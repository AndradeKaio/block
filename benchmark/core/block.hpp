#pragma once

#include <optional>
#include <vector>

namespace benchmark_core {

struct Block {
  int r = 0;
  int c = 0;
  int h = 0;
  int w = 0;

  long long imperfections = 0;

  long long offset = 0;

  long long num_cells() const;
  long long num_nonzeros() const;

  int r_start() const { return r; }
  int r_end() const { return r + h - 1; }
  int c_start() const { return c; }
  int c_end() const { return c + w - 1; }
};

long long assign_offsets(std::vector<Block> &blocks);

struct KRange {
  int k0 = 0;
  int k1 = 0;
};

struct Intersection {
  KRange k;
  Block output;
};

std::optional<Intersection> intersect(const Block &a, const Block &b);

} // namespace benchmark_core
