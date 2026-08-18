#!/usr/bin/env python3
"""
mine_matrices.py — Download, reorder, and mine block patterns from SuiteSparse matrices.

Params are passed as CLI args (or via --env in docker run):

  python mine_matrices.py \\
    --output-dir /data \\
    --kind structural \\
    --min-rows 50000 --max-rows 600000 \\
    --min-nnz 500000 --max-nnz 30000000 \\
    --limit 10 \\
    --workers 4 \\
    --twf 0.5 --to 0.3 --thslim 50 \\
    --small-threshold 10

Docker:
  docker run --rm -v /host/data:/data miner [same args]

Fault tolerance:
  Progress is stored in {output_dir}/progress.db (SQLite).
  Re-running the script skips already-completed matrices.
  Matrices stuck in 'running' state (from a crash) are reset to 'pending'.

Output per matrix (under {output_dir}/{group}/{name}/):
  original_patterns.pkl   — list of BlockPattern objects
  summary.json            — scalar stats
  (the .mtx file is deleted after loading to save disk space)
"""

import argparse
import contextlib
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from pathlib import Path
from threading import Lock

from tqdm import tqdm
from pqdm.threads import pqdm

try:
    import ssgetpy
except ImportError:
    sys.exit("ssgetpy not installed. Run: pip install ssgetpy")


# ── C++ miner binary ─────────────────────────────────────────────────────────

def _build_miner(mining_dir: Path) -> Path:
    """Compile mine_matrix.cpp next to the script. Returns the binary path."""
    src = mining_dir / "mine_matrix.cpp"
    out = mining_dir / "mine_matrix"
    core_dir = mining_dir.parent / "core"

    if not src.exists():
        raise RuntimeError(f"mine_matrix.cpp not found in {mining_dir}")
    if not core_dir.is_dir():
        raise RuntimeError(f"core/ directory not found at {core_dir}")

    # Prefer hdf5-serial (Debian/Ubuntu); fall back to plain hdf5
    hdf5_cflags, hdf5_libs = [], []
    for pkg in ("hdf5-serial", "hdf5"):
        r = subprocess.run(["pkg-config", "--cflags", "--libs", pkg],
                           capture_output=True, text=True)
        if r.returncode == 0:
            hdf5_cflags = r.stdout.split()
            hdf5_libs   = r.stdout.split()
            break

    core_srcs = [
        str(core_dir / f)
        for f in ("block.cpp", "block_generator.cpp", "block_mining.cpp",
                  "interval_tree.cpp", "matrix.cpp", "matrix_io.cpp",
                  "segment_tree.cpp")
    ]
    cmd = [
        "g++", "-O3", "-std=c++20",
        f"-I{core_dir}",
        str(src), *core_srcs,
        "-DHAVE_HDF5", *hdf5_cflags,
        "-o", str(out),
    ]
    print(f"Building mine_matrix … ", end="", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"Compilation failed:\n{r.stderr}")
    print("ok")
    return out


def _locate_miner() -> str:
    mining_dir = Path(__file__).parent
    # Already built next to this script
    candidate = mining_dir / "mine_matrix"
    if candidate.exists():
        return str(candidate)
    # On PATH
    found = shutil.which("mine_matrix")
    if found:
        return found
    # Source present — build automatically
    if (mining_dir / "mine_matrix.cpp").exists():
        return str(_build_miner(mining_dir))
    raise RuntimeError(
        "mine_matrix binary not found and mine_matrix.cpp is missing.\n"
        "Ensure mine_matrix.cpp is in the same directory as this script."
    )


# ── SQLite state store ────────────────────────────────────────────────────────

class StateDB:
    """Thread-safe SQLite wrapper for tracking matrix processing state."""

    def __init__(self, path: Path):
        self.path = str(path)
        self._lock = Lock()
        self._init()

    def _connect(self):
        return sqlite3.connect(self.path, check_same_thread=False)

    def _init(self):
        with self._lock:
            con = self._connect()
            con.executescript("""
                CREATE TABLE IF NOT EXISTS matrices (
                    id          INTEGER PRIMARY KEY,
                    name        TEXT,
                    grp         TEXT,
                    rows        INTEGER,
                    cols        INTEGER,
                    nnz         INTEGER,
                    kind        TEXT,
                    status      TEXT DEFAULT 'pending',
                    error       TEXT,
                    started_at  REAL,
                    finished_at REAL
                );
                CREATE TABLE IF NOT EXISTS results (
                    matrix_id       INTEGER,
                    ordering        TEXT,
                    n_patterns      INTEGER,
                    n_large         INTEGER,
                    max_nnz         INTEGER,
                    mean_nnz        REAL,
                    dominant_shape  TEXT,
                    dominant_count  INTEGER,
                    dominant_share  REAL,
                    mining_time     REAL,
                    PRIMARY KEY (matrix_id, ordering)
                );
            """)
            # Reset any 'running' rows left from a previous crashed run
            con.execute("UPDATE matrices SET status='pending' WHERE status='running'")
            # Migrate: add columns if missing (backward-compatible)
            for col in ("padding_zeros INTEGER", "covered_nnz INTEGER",
                        "total_padding INTEGER", "n_singles INTEGER"):
                try:
                    con.execute(f"ALTER TABLE results ADD COLUMN {col}")
                except sqlite3.OperationalError:
                    pass
            con.commit()
            con.close()

    def upsert_matrices(self, matrix_list):
        rows = [
            (m.id, m.name, m.group, m.rows, m.cols, m.nnz, m.kind)
            for m in matrix_list
        ]
        with self._lock:
            con = self._connect()
            con.executemany("""
                INSERT OR IGNORE INTO matrices(id, name, grp, rows, cols, nnz, kind)
                VALUES (?,?,?,?,?,?,?)
            """, rows)
            con.commit()
            con.close()

    def get_pending(self, include_done: bool = False):
        statuses = "('pending', 'failed', 'done')" if include_done else "('pending', 'failed')"
        with self._lock:
            con = self._connect()
            rows = con.execute(
                f"SELECT id, name, grp, rows, nnz FROM matrices WHERE status IN {statuses}"
            ).fetchall()
            con.close()
        return rows

    def reset_done_to_pending(self, matrix_ids):
        with self._lock:
            con = self._connect()
            con.executemany(
                "UPDATE matrices SET status='pending' WHERE id=? AND status='done'",
                [(mid,) for mid in matrix_ids]
            )
            con.commit()
            con.close()

    def mark_running(self, matrix_id):
        with self._lock:
            con = self._connect()
            con.execute(
                "UPDATE matrices SET status='running', started_at=? WHERE id=?",
                (time.time(), matrix_id)
            )
            con.commit()
            con.close()

    def mark_done(self, matrix_id):
        with self._lock:
            con = self._connect()
            con.execute(
                "UPDATE matrices SET status='done', finished_at=? WHERE id=?",
                (time.time(), matrix_id)
            )
            con.commit()
            con.close()

    def mark_no_patterns(self, matrix_id):
        with self._lock:
            con = self._connect()
            con.execute(
                "UPDATE matrices SET status='no_patterns', finished_at=? WHERE id=?",
                (time.time(), matrix_id)
            )
            con.commit()
            con.close()

    def mark_failed(self, matrix_id, error: str):
        with self._lock:
            con = self._connect()
            con.execute(
                "UPDATE matrices SET status='failed', error=?, finished_at=? WHERE id=?",
                (error[:2000], time.time(), matrix_id)
            )
            con.commit()
            con.close()

    def save_result(self, matrix_id, ordering, stats, mining_time):
        with self._lock:
            con = self._connect()
            con.execute("""
                INSERT OR REPLACE INTO results
                  (matrix_id, ordering, n_patterns, n_large, max_nnz,
                   mean_nnz, dominant_shape, dominant_count, dominant_share,
                   mining_time, padding_zeros, covered_nnz, total_padding,
                   n_singles)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                matrix_id, ordering,
                stats["n_patterns"], stats["n_large"], stats["max_nnz"],
                stats["mean_nnz"], stats["dominant_shape"],
                stats["dominant_count"], stats["dominant_share"],
                mining_time,
                stats.get("padding_zeros"), stats.get("covered_nnz"),
                stats.get("total_padding"), stats.get("n_singles"),
            ))
            con.commit()
            con.close()

    def summary(self):
        with self._lock:
            con = self._connect()
            rows = con.execute("""
                SELECT status, COUNT(*) FROM matrices GROUP BY status
            """).fetchall()
            con.close()
        return dict(rows)


# ── Per-matrix worker ─────────────────────────────────────────────────────────

@contextlib.contextmanager
def _quiet_stderr():
    """Suppress stderr (used to hide ssgetpy's per-file tqdm download bars)."""
    devnull = open(os.devnull, "w")
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stderr = sys.__stderr__  # always restore to original, never a closed devnull
        devnull.close()


def process_matrix(task):
    m, args, db, out_root, nnz_counter, nnz_lock = task
    mat_dir = out_root / m.group / m.name
    mat_dir.mkdir(parents=True, exist_ok=True)

    result = {"id": m.id, "name": m.name, "status": "ok"}

    tqdm.write(f"[START] {m.group}/{m.name}  ({m.nnz:,} nnz)  →  {mat_dir}")

    try:
        db.mark_running(m.id)

        # ── 1. Locate / download MTX ──────────────────────────────────
        mtx_path = mat_dir / f"{m.name}.mtx"
        if not mtx_path.exists():
            tqdm.write(f"  [DL]  {m.name}.mtx")
            with _quiet_stderr():
                m.download(format="MM", destpath=str(mat_dir), extract=True)
            candidates = list(mat_dir.rglob("*.mtx"))
            if not candidates:
                raise FileNotFoundError(f"No .mtx found after download in {mat_dir}")
            if candidates[0] != mtx_path:
                leftover_dir = candidates[0].parent
                candidates[0].rename(mtx_path)
                # ssgetpy creates its own name/ (or group/name/) subdirectory
                # inside mat_dir; remove those empty directories now.
                for d in [leftover_dir, leftover_dir.parent]:
                    if d != mat_dir:
                        try:
                            d.rmdir()
                        except OSError:
                            break
        else:
            tqdm.write(f"  [HIT] {m.name}.mtx already present")

        # ── 2. Mine + write .bsp via C++ executable ───────────────────
        bsp_path = mat_dir / f"{m.name}.bsp"
        miner_cmd = [
            _locate_miner(),
            str(mtx_path),
            str(bsp_path),
            "--twf",             str(args.twf),
            "--to",              str(args.to),
            "--thslim",          str(args.thslim),
            "--small-threshold", str(args.small_threshold),
        ]
        tqdm.write(f"  [MINE] {' '.join(miner_cmd)}")
        t0 = time.time()
        proc = subprocess.run(miner_cmd, capture_output=True, text=True, timeout=600)
        orig_time = time.time() - t0

        if proc.stderr.strip():
            tqdm.write(f"  [WARN] {proc.stderr.strip()}")

        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or f"mine_matrix exited {proc.returncode}")

        orig_stats = json.loads(proc.stdout)
        tqdm.write(f"  [MINE] done in {orig_time:.1f}s  →  {bsp_path}")

        # ── 3. No-patterns check ──────────────────────────────────────
        if orig_stats["n_large"] == 0:
            db.mark_no_patterns(m.id)
            result["status"] = "no_patterns"
            tqdm.write(f"[SKIP] {m.name}: no large patterns")
            return result

        # ── 4. Persist to DB ──────────────────────────────────────────
        db.save_result(m.id, "original", orig_stats, orig_time)

        # ── 5. Save per-matrix summary ────────────────────────────────
        summary = {
            "id": m.id, "name": m.name, "group": m.group,
            "rows": m.rows, "cols": m.cols, "nnz": m.nnz,
            "original": {**orig_stats, "mining_time_s": round(orig_time, 3)},
        }
        with open(mat_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        db.mark_done(m.id)
        result["orig_patterns"] = orig_stats["n_large"]
        tqdm.write(f"[DONE] {m.name}: {orig_stats['n_large']} patterns, dominant {orig_stats['dominant_shape']}")

    except Exception as e:
        err = traceback.format_exc()
        db.mark_failed(m.id, err)
        result["status"] = "failed"
        result["error"]  = str(e)
        tqdm.write(f"[FAIL] {m.name}: {e}")

    finally:
        with nnz_lock:
            nnz_counter[0] += m.nnz

    return result


# ── Progress display ──────────────────────────────────────────────────────────

class DualProgress:
    """Two tqdm bars: one per-matrix, one NNZ-weighted."""

    def __init__(self, n_matrices, total_nnz):
        self.mat_bar = tqdm(
            total=n_matrices, desc="Matrices  ",
            unit="mat", position=0, colour="cyan",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
        )
        self.nnz_bar = tqdm(
            total=total_nnz, desc="Work (nnz)",
            unit_scale=True, unit="nnz", position=1, colour="green",
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}  {percentage:.1f}%",
        )
        self._lock = Lock()

    def update(self, nnz_done):
        with self._lock:
            self.mat_bar.update(1)
            self.nnz_bar.update(nnz_done)

    def close(self):
        self.mat_bar.close()
        self.nnz_bar.close()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Mine block patterns from SuiteSparse matrices.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Search filters
    p.add_argument("--kind", nargs="*", default=["structural"], metavar="KIND",
                   help="Matrix kind filter (one or more of: structural, real, integer, complex). "
                        "Pass --kind with no value to search all kinds.")
    p.add_argument("--min-rows",  type=int, default=None, dest="min_rows")
    p.add_argument("--max-rows",  type=int, default=None, dest="max_rows")
    p.add_argument("--min-nnz",   type=int, default=None, dest="min_nnz")
    p.add_argument("--max-nnz",   type=int, default=None, dest="max_nnz")
    p.add_argument("--limit",     type=int, default=10,   help="Max matrices to process")
    p.add_argument("--matrix-ids",nargs="*", type=int, default=None, dest="matrix_ids",
                   help="Explicit SuiteSparse IDs (overrides search filters)")
    # Execution
    p.add_argument("--workers",   type=int, default=4,    help="Parallel worker threads")
    p.add_argument("--output-dir",
                   default="/data",
                   dest="output_dir",
                   help="Root directory for downloads and results")
    p.add_argument("--force", action="store_true", default=False,
                   help="Re-mine matrices that were already processed locally")
    # Mining params
    p.add_argument("--twf",             type=float, default=0.5)
    p.add_argument("--to",              type=float, default=0.3)
    p.add_argument("--thslim",          type=int,   default=50)
    p.add_argument("--small-threshold", type=int,   default=10, dest="small_threshold")
    p.add_argument("--retry-expand",    action="store_true", default=True, dest="retry_expand",
                   help="Use retry expand strategy — default, kept for explicitness")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"Output directory : {out_root.resolve()}")
    print(f"Progress DB      : {out_root / 'progress.db'}")
    print(f"Miner binary     : {_locate_miner()}")

    db = StateDB(out_root / "progress.db")

    # ── Resolve matrix list ───────────────────────────────────────────
    print("Querying SuiteSparse index ...")
    if args.matrix_ids:
        matrices = [ssgetpy.search(matid=mid)[0] for mid in args.matrix_ids]
    else:
        row_bounds = (args.min_rows, args.max_rows) if (args.min_rows or args.max_rows) else None
        nnz_bounds = (args.min_nnz,  args.max_nnz)  if (args.min_nnz  or args.max_nnz)  else None
        seen_ids = set()
        matrices = []
        kinds = args.kind if args.kind else [None]  # empty list → one pass with no kind filter
        for kind in kinds:
            search_kwargs = dict(rowbounds=row_bounds, nzbounds=nnz_bounds, limit=args.limit)
            if kind is not None:
                search_kwargs["kind"] = kind
            for m in ssgetpy.search(**search_kwargs):
                if m.id not in seen_ids:
                    seen_ids.add(m.id)
                    matrices.append(m)

    if not matrices:
        sys.exit("No matrices found matching the given filters.")

    db.upsert_matrices(matrices)

    # ── Filter already-done ───────────────────────────────────────────
    if args.force:
        db.reset_done_to_pending([m.id for m in matrices])
    pending_ids = {row[0] for row in db.get_pending()}
    todo = [m for m in matrices if m.id in pending_ids]
    skipped = len(matrices) - len(todo)

    if not todo:
        print(f"Nothing to do ({skipped} already completed).")
        return

    total_nnz = sum(m.nnz for m in todo)
    print(
        f"{len(todo)} matrices to process"
        + (f"  ({skipped} skipped)" if skipped else "")
        + f"  |  nnz = {total_nnz:,}  |  workers = {args.workers}"
    )

    # ── Shared NNZ counter ────────────────────────────────────────────
    from threading import Lock as TLock
    nnz_counter = [0]
    nnz_lock    = TLock()
    progress    = DualProgress(len(todo), total_nnz)

    # ── Build tasks ───────────────────────────────────────────────────
    tasks = [
        (m, args, db, out_root, nnz_counter, nnz_lock)
        for m in todo
    ]

    # ── pqdm with post-update hook ────────────────────────────────────
    # pqdm handles the futures; we update our dual bars in a wrapper.
    def worker(task):
        result = process_matrix(task)
        progress.update(task[0].nnz)
        return result

    results = pqdm(
        tasks,
        worker,
        n_jobs=args.workers,
        argument_type=None,
        exception_behaviour="deferred",
        desc="",          # we manage our own bars
        disable=True,     # suppress pqdm's own bar; ours handle display
    )

    progress.close()

    # ── Final summary (bars are closed, safe to print) ────────────────
    counts  = db.summary()
    done    = counts.get("done", 0)
    failed  = counts.get("failed", 0)
    total   = sum(counts.values())

    print(f"\n{done}/{total} completed  |  {failed} failed  |  {out_root / 'progress.db'}")

    fail_results = [r for r in results if isinstance(r, dict) and r.get("status") == "failed"]
    if fail_results:
        print("Failed:")
        for r in fail_results:
            print(f"  {r['name']}: {r['error']}")


if __name__ == "__main__":
    main()
