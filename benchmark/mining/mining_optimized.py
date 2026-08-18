"""
Optimized block pattern mining on RCM-reordered sparse matrix.

Changes vs. original:
  - CSRContainer replaces heap + Python dict:
      * numpy bool alive-mask over CSR indices (builds in ~4ms vs ~17s)
      * vectorized row queries: searchsorted + slice sum
      * vectorized block deletion: slice assignment
      * row-pointer scan for get_start_point (no heap overhead)
  - expand_block avoids creating BlockPattern objects during expansion;
    only constructs the final one
  - Bounds guarded in query_col / query_row
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
from scipy.sparse.csgraph import reverse_cuthill_mckee
from dataclasses import dataclass
import time
import sys


# ── Data structures (unchanged) ──────────────────────────────────────────

@dataclass
class BlockPattern:
    r: int; c: int; h: int; w: int; imperfections: int = 0

    def num_cells(self):          return self.h * self.w
    def num_nonzeros(self):       return self.num_cells() - self.imperfections
    def imperfection_ratio(self): return self.imperfections / self.num_cells() if self.num_cells() else 0
    def aspect_ratio(self):       return max(self.h, self.w) / min(self.h, self.w)
    def score(self, penalty=1.5): return self.num_nonzeros() - self.imperfections * penalty
    def elements(self):
        for i in range(self.h):
            for j in range(self.w):
                yield (self.r + i, self.c + j)


# ── CSR container ─────────────────────────────────────────────────────────

class CSRContainer:
    """
    Sparse matrix backed by CSR arrays + a boolean alive mask.

    Replaces the original heap + Python dict, which costs ~17s and ~2 GB
    just to build for a 20 M-entry matrix. This builds in ~4 ms.

    Row queries (down-expansion) are vectorized via numpy searchsorted.
    Column queries (right-expansion) loop over rows but use binary search
    into numpy arrays rather than Python dict lookups.
    Block deletion is a vectorized slice assignment per row.
    """

    def __init__(self, A: sp.csr_matrix):
        A.sort_indices()
        self.n       = A.shape[0]
        self.indptr  = A.indptr           # shape (n+1,)
        self.indices = A.indices          # shape (nnz,), sorted within each row
        self.alive   = np.ones(A.nnz, dtype=np.bool_)
        self._count  = int(A.nnz)
        self._row    = 0                  # row-pointer for get_start_point

    def __len__(self):  return self._count
    def __bool__(self): return self._count > 0

    # ── Membership ───────────────────────────────────────────────────────

    def _find(self, row, col):
        rs = int(self.indptr[row]); re = int(self.indptr[row + 1])
        if rs == re: return -1
        k = int(np.searchsorted(self.indices[rs:re], col))
        idx = rs + k
        return idx if k < (re - rs) and self.indices[idx] == col else -1

    def __contains__(self, pos):
        k = self._find(pos[0], pos[1])
        return k >= 0 and bool(self.alive[k])

    # ── Queries ──────────────────────────────────────────────────────────

    def query_row(self, row, col_lo, col_hi):
        """Count alive entries in `row` for columns in [col_lo, col_hi)."""
        if row >= self.n: return 0
        rs = int(self.indptr[row]); re = int(self.indptr[row + 1])
        if rs == re: return 0
        cols = self.indices[rs:re]
        lo = int(np.searchsorted(cols, col_lo))
        hi = int(np.searchsorted(cols, col_hi))
        return int(self.alive[rs + lo: rs + hi].sum()) if lo < hi else 0

    def query_col(self, col, row_lo, row_hi):
        """Count alive entries in `col` for rows in [row_lo, row_hi)."""
        count = 0
        row_hi = min(row_hi, self.n)
        for row in range(row_lo, row_hi):
            rs = int(self.indptr[row]); re = int(self.indptr[row + 1])
            if rs == re: continue
            k = int(np.searchsorted(self.indices[rs:re], col))
            if k < (re - rs) and self.indices[rs + k] == col and self.alive[rs + k]:
                count += 1
        return count

    # ── Deletion ─────────────────────────────────────────────────────────

    def delete_block(self, r, c, h, w):
        """Mark all entries in the h×w block at (r, c) as dead."""
        for row in range(r, min(r + h, self.n)):
            rs = int(self.indptr[row]); re = int(self.indptr[row + 1])
            if rs == re: continue
            cols = self.indices[rs:re]
            lo = int(np.searchsorted(cols, c))
            hi = int(np.searchsorted(cols, c + w))
            if lo < hi:
                sl = self.alive[rs + lo: rs + hi]
                self._count -= int(sl.sum())
                sl[:] = False

    # ── Next seed ────────────────────────────────────────────────────────

    def get_start_point(self):
        """Return the first alive (row, col) in row-major order, or None."""
        while self._row < self.n:
            rs = int(self.indptr[self._row]); re = int(self.indptr[self._row + 1])
            if rs < re:
                alive_row = self.alive[rs:re]
                if alive_row.any():
                    k = int(np.argmax(alive_row))
                    return (self._row, int(self.indices[rs + k]))
            self._row += 1
        return None


# ── Block expansion ───────────────────────────────────────────────────────

def expand_block(seed, container, Twf=0.5, To=0.3, Thslim=50):
    r, c = seed
    h = w = 1
    imp = 0  # seed is always alive (guaranteed by get_start_point)

    expand_right = expand_down = True

    while expand_right or expand_down:
        moved = False

        # ── Right ─────────────────────────────────────────────────────
        if expand_right:
            nw = w + 1
            if max(h, nw) / min(h, nw) > Thslim and nw > h:
                expand_right = False
            else:
                count = container.query_col(c + w, r, r + h)
                if count >= h * (1.0 - Twf):
                    new_imp = imp + (h - count)
                    if new_imp / (h * nw) <= To:
                        w = nw; imp = new_imp; moved = True
                    else:
                        expand_right = False
                else:
                    expand_right = False

        # ── Down ──────────────────────────────────────────────────────
        if expand_down:
            nh = h + 1
            if max(nh, w) / min(nh, w) > Thslim and nh > w:
                expand_down = False
            else:
                count = container.query_row(r + h, c, c + w)
                if count >= w * (1.0 - Twf):
                    new_imp = imp + (w - count)
                    if new_imp / (nh * w) <= To:
                        h = nh; imp = new_imp; moved = True
                    else:
                        expand_down = False
                else:
                    expand_down = False

        if not moved:
            break

    return BlockPattern(r=r, c=c, h=h, w=w, imperfections=imp)


# ── Composite mining (unchanged logic) ───────────────────────────────────

def build_composite(small_patterns, Thmin=3, Thsize=10):
    def shape_key(p):
        if isinstance(p, BlockPattern): return ("block", p.h, p.w)
        return ("unknown",)

    groups = {}
    for p in small_patterns:
        groups.setdefault(shape_key(p), []).append(p)

    composite, used = [], set()
    for key, group in groups.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if id(group[i]) in used or id(group[j]) in used: continue
                p0, p1 = group[i], group[j]
                dr, dc = p1.r - p0.r, p1.c - p0.c
                chain = [p0, p1]
                for k in range(len(group)):
                    pk = group[k]
                    if id(pk) in used or pk in chain: continue
                    prev = chain[-1]
                    if pk.r - prev.r == dr and pk.c - prev.c == dc:
                        chain.append(pk)
                total_nnz = sum(p.num_nonzeros() for p in chain)
                if len(chain) >= Thmin and total_nnz >= Thsize:
                    composite.append(chain)
                    for p in chain: used.add(id(p))

    unused = [p for p in small_patterns if id(p) not in used]
    return composite, unused


# ── Main mining loop ──────────────────────────────────────────────────────

def mine_patterns(A_csr, Twf=0.5, To=0.3, Thslim=50,
                  small_threshold=10, verysmall_threshold=3, penalty=1.5):
    container = CSRContainer(A_csr)
    total     = len(container)
    all_basic = []
    t0        = time.time()
    n_iter    = 0

    while container:
        start = container.get_start_point()
        if start is None: break

        block = expand_block(start, container, Twf, To, Thslim)
        all_basic.append(block)
        container.delete_block(block.r, block.c, block.h, block.w)

        n_iter += 1
        if n_iter % 25_000 == 0:
            remaining = len(container)
            elapsed   = time.time() - t0
            done      = 1 - remaining / total
            eta       = (elapsed / done - elapsed) if done > 0 else 0
            print(f"  [{elapsed:6.1f}s] iter={n_iter:,}  patterns={len(all_basic):,}  "
                  f"remaining={remaining:,}  ETA={eta:.0f}s")

    large_p = [p for p in all_basic if p.num_nonzeros() >= small_threshold]
    small_p  = [p for p in all_basic if p.num_nonzeros() <  small_threshold]
    composite_p, unused_p = build_composite(small_p)
    leftover = [p for p in unused_p if p.num_nonzeros() >= verysmall_threshold]

    return {"basic": large_p, "composite": composite_p,
            "small": leftover, "nonaffine": None}


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    t_start = time.time()

    print("Loading msdoor.mtx ...")
    A = sio.mmread("/workspace/msdoor.mtx")
    A = sp.csr_matrix(A)
    print(f"  shape={A.shape}  nnz={A.nnz:,}")

    print("RCM reordering ...")
    perm = reverse_cuthill_mckee(A, symmetric_mode=True)
    A = A[perm][:, perm]
    A.sort_indices()
    print(f"  done  ({time.time()-t_start:.1f}s total so far)")

    print("Mining patterns ...")
    t_mine = time.time()
    results = mine_patterns(A)
    t_done  = time.time()
    print(f"Mining done in {t_done - t_mine:.1f}s")

    # ── Summary ───────────────────────────────────────────────────────────
    basic     = results["basic"]
    composite = results["composite"]
    small     = results["small"]

    nnz_total = A.nnz

    print(f"\n{'='*60}")
    print(f"RESULTS  (total matrix nnz = {nnz_total:,})")
    print(f"{'='*60}")
    print(f"Large patterns  (nnz >= 10) : {len(basic):,}")
    if basic:
        nnz_list   = [p.num_nonzeros() for p in basic]
        covered    = sum(nnz_list)
        print(f"  largest block nnz : {max(nnz_list)}")
        print(f"  mean block nnz    : {np.mean(nnz_list):.1f}")
        print(f"  total nnz covered : {covered:,}  ({100*covered/nnz_total:.1f}%)")

        top = sorted(basic, key=lambda p: p.num_nonzeros(), reverse=True)[:10]
        print(f"\n  Top 10 blocks by nnz:")
        print(f"  {'(r,c)':>20}  {'h':>6}  {'w':>6}  {'nnz':>8}  {'imp':>5}  imp%")
        for p in top:
            print(f"  ({p.r:>8},{p.c:>8})  {p.h:>6}  {p.w:>6}  "
                  f"{p.num_nonzeros():>8}  {p.imperfections:>5}  "
                  f"{100*p.imperfection_ratio():.1f}%")

        # Block size distribution
        size_buckets = {}
        for p in basic:
            bucket = max(p.h, p.w)
            bucket = (bucket // 10) * 10
            size_buckets[bucket] = size_buckets.get(bucket, 0) + 1
        print(f"\n  Block size distribution (by max(h,w)):")
        for k in sorted(size_buckets):
            print(f"    {k:>4}+  : {size_buckets[k]:,}")

    print(f"\nComposite groups  : {len(composite):,}")
    print(f"Leftover small    : {len(small):,}")
    print(f"\nTotal wall time   : {t_done - t_start:.1f}s")

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = "/tmp/patterns_rcm.txt"
    with open(out_path, "w") as f:
        f.write("# basic patterns\n")
        for p in basic:
            f.write(f"{p}\n")
        f.write("\n# composite\n")
        for chain in composite:
            for p in chain:
                f.write(f"{p}\n")
            f.write("---\n")
    print(f"Patterns saved to {out_path}")
