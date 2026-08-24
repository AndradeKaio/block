#include "pipeline.hpp"

#include <algorithm>
#include <chrono>
#include <numeric>
#include <set>
#include <unordered_map>

#include "interval_tree.hpp"
#include "segment_tree.hpp"

#ifdef _OPENMP
#include <omp.h>
#else
inline int omp_get_max_threads() { return 1; }
inline int omp_get_thread_num()  { return 0; }
#endif

namespace benchmark_core {

IntersectionResult find_intersecting_pairs(const std::vector<Block> &a_blocks,
                                           const std::vector<Block> &b_blocks,
                                           IntersectionTimings *timings) {
  using Clock = std::chrono::steady_clock;
  auto elapsed_ms = [](Clock::time_point t0) {
    return float(std::chrono::duration<double, std::milli>(Clock::now() - t0).count());
  };

  // Build interval tree over B's row ranges (sequential — dependent inserts).
  auto t_build = Clock::now();
  IntervalTree b_tree;
  for (int bi = 0; bi < static_cast<int>(b_blocks.size()); ++bi) {
    const Block &b = b_blocks[bi];
    b_tree.insert(b.r_start(), b.r_end() + 1, bi);
  }
  const float tree_build_ms = elapsed_ms(t_build);

  // Query the (now immutable) B-tree for each A-block in parallel.
  // IntervalTree::query() is a pure read traversal — safe for concurrent use.
  auto t_query = Clock::now();
  const int nA = static_cast<int>(a_blocks.size());
  const int nT = omp_get_max_threads();
  std::vector<IntersectionResult> per_thread(nT);

#pragma omp parallel for schedule(dynamic, 64)
  for (int ai = 0; ai < nA; ++ai) {
    IntersectionResult &r = per_thread[omp_get_thread_num()];
    const Block &a = a_blocks[ai];
    auto hits = b_tree.query(a.c_start(), a.c_end() + 1);
    std::sort(hits.begin(), hits.end(),
              [](const IntervalEntry &x, const IntervalEntry &y) {
                return x.data < y.data;
              });
    for (const auto &hit : hits) {
      const int bi = hit.data;
      auto isec = intersect(a, b_blocks[bi]);
      if (isec) {
        r.contributions.push_back({ai, bi, isec->k});
        r.output_blocks.push_back(isec->output);
      }
    }
  }
  const float query_ms = elapsed_ms(t_query);

  if (timings) {
    timings->tree_build_ms = tree_build_ms;
    timings->query_ms      = query_ms;
  }

  // Merge per-thread results. Ordering is nondeterministic across thread
  // counts; merge_overlapping_output_blocks is invariant to it.
  IntersectionResult result;
  for (auto &r : per_thread) {
    result.contributions.insert(result.contributions.end(),
                                r.contributions.begin(), r.contributions.end());
    result.output_blocks.insert(result.output_blocks.end(),
                                r.output_blocks.begin(), r.output_blocks.end());
  }
  return result;
}

std::vector<Group>
merge_overlapping_output_blocks(const std::vector<Block> &output_blocks) {
  const int n = static_cast<int>(output_blocks.size());

  // Row-panel output_blocks directly (their row range IS their contributing
  // A-block's row range, see intersect() in block.cpp) -- same sweep as
  // prisma_cpu_spmm_bench.cpp's row_groups: sort by start row, merge while a
  // block's start row falls inside the current panel's row extent.
  std::vector<int> order(n);
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int x, int y) {
    return output_blocks[x].r_start() < output_blocks[y].r_start();
  });

  std::vector<std::vector<int>> panels; // panels[p] = global output_blocks indices
  int row_end = -1;
  for (int i : order) {
    const Block &b = output_blocks[i];
    if (b.r_start() >= row_end) {
      panels.push_back({});
      row_end = b.r_end() + 1;
    } else {
      row_end = std::max(row_end, b.r_end() + 1);
    }
    panels.back().push_back(i);
  }

  const int np = static_cast<int>(panels.size());
  std::vector<std::vector<Group>> per_panel(np);

  // Panels are row-disjoint by construction, so their sweeps touch entirely
  // separate IntervalTree/union-find state -- safe to run fully in parallel.
#pragma omp parallel for schedule(dynamic, 1)
  for (int p = 0; p < np; ++p) {
    const std::vector<int> &idxs = panels[p];
    const int m = static_cast<int>(idxs.size());

    std::vector<int> parent(m);
    std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int x) {
      while (parent[x] != x) {
        parent[x] = parent[parent[x]];
        x = parent[x];
      }
      return x;
    };
    auto unite = [&](int x, int y) {
      const int rx = find(x);
      const int ry = find(y);
      if (rx != ry)
        parent[rx] = ry;
    };

    enum EventKind { kEnd = 0, kStart = 1 };
    struct Event {
      int pos;
      EventKind kind;
      int local_idx;
    };
    std::vector<Event> events;
    events.reserve(m * 2);
    for (int li = 0; li < m; ++li) {
      const Block &b = output_blocks[idxs[li]];
      events.push_back({b.r_start(), kStart, li});
      events.push_back({b.r_end() + 1, kEnd, li});
    }
    std::sort(events.begin(), events.end(), [](const Event &x, const Event &y) {
      if (x.pos != y.pos)
        return x.pos < y.pos;
      return x.kind < y.kind;
    });

    IntervalTree active;
    std::vector<IntervalEntry> hits;
    for (const auto &e : events) {
      const Block &b = output_blocks[idxs[e.local_idx]];
      if (e.kind == kStart) {
        hits.clear();
        active.query(b.c_start(), b.c_end() + 1, hits);
        for (const auto &hit : hits)
          unite(e.local_idx, hit.data);
        active.insert(b.c_start(), b.c_end() + 1, e.local_idx);
      } else {
        active.remove(b.c_start(), b.c_end() + 1, e.local_idx);
      }
    }

    std::vector<Group> &out = per_panel[p];
    std::unordered_map<int, std::size_t> group_index;
    for (int li = 0; li < m; ++li) {
      const int root = find(li);
      auto it = group_index.find(root);
      if (it == group_index.end()) {
        group_index.emplace(root, out.size());
        out.push_back(Group{root, {idxs[li]}}); // idxs[li]: back to global index
      } else {
        out[it->second].members.push_back(idxs[li]);
      }
    }
  }

  std::vector<Group> groups;
  groups.reserve(n);
  for (int p = 0; p < np; ++p) {
    for (auto &g : per_panel[p]) {
      // Local roots are only meaningful within their own panel's
      // union-find; renumber to a globally unique id (nothing downstream
      // depends on the specific value, only that members sharing a group
      // share an id).
      g.id = static_cast<int>(groups.size());
      groups.push_back(std::move(g));
    }
  }
  return groups;
}

std::vector<std::pair<int, int>>
find_overlapping_output_blocks(const std::vector<Block> &output_blocks) {
  enum EventKind { kEnd = 0, kStart = 1 };
  struct Event {
    int pos;
    EventKind kind;
    int idx;
  };

  std::vector<Event> events;
  events.reserve(output_blocks.size() * 2);
  for (int idx = 0; idx < static_cast<int>(output_blocks.size()); ++idx) {
    const Block &b = output_blocks[idx];
    events.push_back({b.r_start(), kStart, idx});
    events.push_back({b.r_end() + 1, kEnd, idx});
  }
  // End events sort before start events at the same position, so a block
  // that ends exactly where another begins is not counted as overlapping.
  std::sort(events.begin(), events.end(), [](const Event &x, const Event &y) {
    if (x.pos != y.pos)
      return x.pos < y.pos;
    return x.kind < y.kind;
  });

  IntervalTree active;
  std::set<std::pair<int, int>> pairs;

  for (const auto &e : events) {
    const Block &b = output_blocks[e.idx];
    if (e.kind == kStart) {
      for (const auto &hit : active.query(b.c_start(), b.c_end() + 1)) {
        int i = e.idx;
        int j = hit.data;
        if (i > j)
          std::swap(i, j);
        pairs.emplace(i, j);
      }
      active.insert(b.c_start(), b.c_end() + 1, e.idx);
    } else {
      active.remove(b.c_start(), b.c_end() + 1, e.idx);
    }
  }
  return std::vector<std::pair<int, int>>(pairs.begin(), pairs.end());
}

std::vector<Group>
merge_groups(const std::vector<Block> &output_blocks,
             const std::vector<std::pair<int, int>> &overlap_pairs) {
  const int n = static_cast<int>(output_blocks.size());
  std::vector<int> parent(n);
  std::iota(parent.begin(), parent.end(), 0);

  auto find = [&](int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };
  auto unite = [&](int x, int y) {
    const int rx = find(x);
    const int ry = find(y);
    if (rx != ry) {
      parent[rx] = ry;
    }
  };

  for (const auto &[x, y] : overlap_pairs) {
    unite(x, y);
  }

  std::vector<Group> groups;
  std::unordered_map<int, std::size_t> group_index;
  for (int i = 0; i < n; ++i) {
    const int root = find(i);
    auto it = group_index.find(root);
    if (it == group_index.end()) {
      group_index.emplace(root, groups.size());
      groups.push_back(Group{root, {i}});
    } else {
      groups[it->second].members.push_back(i);
    }
  }
  return groups;
}

Block mbr(const std::vector<Block> &blocks) {
  std::vector<int> ys;
  ys.reserve(blocks.size() * 2);
  for (const auto &b : blocks) {
    ys.push_back(b.r_start());
    ys.push_back(b.r_end() + 1);
  }
  SegmentTree tree(ys);

  int min_row = blocks.front().r_start();
  int max_row = blocks.front().r_end();
  int min_col = blocks.front().c_start();
  int max_col = blocks.front().c_end();
  for (const auto &b : blocks) {
    min_row = std::min(min_row, b.r_start());
    max_row = std::max(max_row, b.r_end());
    min_col = std::min(min_col, b.c_start());
    max_col = std::max(max_col, b.c_end());
  }

  struct Event {
    int x;
    int t;
    int y1;
    int y2;
  };
  std::vector<Event> events;
  events.reserve(blocks.size() * 2);
  for (const auto &b : blocks) {
    events.push_back({b.c_start(), +1, b.r_start(), b.r_end() + 1});
    events.push_back({b.c_end() + 1, -1, b.r_start(), b.r_end() + 1});
  }
  std::sort(events.begin(), events.end(),
            [](const Event &x, const Event &y) { return x.x < y.x; });

  long long union_area = 0;
  int prev_x = 0;
  for (const auto &e : events) {
    union_area += static_cast<long long>(e.x - prev_x) * tree.total_covered();
    tree.add_interval(e.y1, e.y2, e.t);
    prev_x = e.x;
  }

  Block result;
  result.r = min_row;
  result.c = min_col;
  result.h = max_row - min_row + 1;
  result.w = max_col - min_col + 1;
  result.imperfections = result.num_cells() - union_area;
  return result;
}

FusionResult block_fusion(const std::vector<Block> &output_blocks,
                          const std::vector<Contribution> &contributions,
                          const std::vector<Group> &groups) {
  // Pre-allocate so each thread can write to its own index without locking.
  const int ng = static_cast<int>(groups.size());
  FusionResult result;
  result.fused_contributions.resize(ng);
  result.fused_blocks.resize(ng);

  // Each mbr() call is independent: reads only from output_blocks/contributions
  // (shared, read-only) and writes to its own pre-allocated slot.
#pragma omp parallel for schedule(dynamic, 32)
  for (int gi = 0; gi < ng; ++gi) {
    const auto &group = groups[gi];
    std::vector<Contribution> group_contribs;
    std::vector<Block> group_blocks;
    group_contribs.reserve(group.members.size());
    group_blocks.reserve(group.members.size());
    for (int bid : group.members) {
      group_contribs.push_back(contributions[bid]);
      group_blocks.push_back(output_blocks[bid]);
    }
    result.fused_contributions[gi] = std::move(group_contribs);
    result.fused_blocks[gi] = mbr(group_blocks);
  }
  return result;
}

} // namespace benchmark_core
