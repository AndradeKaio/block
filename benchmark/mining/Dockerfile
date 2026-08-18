# Build with:  docker build -f benchmark/mining/Dockerfile -t miner .
# (run from the benchmark/ directory so both core/ and mining/ are in context)

# ── Stage 1: build mine_matrix ────────────────────────────────────────────────
FROM debian:bookworm-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ libhdf5-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY core/ /src/core/
COPY mining/mine_matrix.cpp /src/

RUN g++ -O3 -std=c++20 \
    -I/src/core \
    /src/mine_matrix.cpp \
    /src/core/block.cpp \
    /src/core/block_generator.cpp \
    /src/core/block_mining.cpp \
    /src/core/interval_tree.cpp \
    /src/core/matrix.cpp \
    /src/core/matrix_io.cpp \
    /src/core/segment_tree.cpp \
    -DHAVE_HDF5 \
    $(pkg-config --cflags --libs hdf5-serial 2>/dev/null || pkg-config --cflags --libs hdf5) \
    -o /usr/local/bin/mine_matrix

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libhdf5-103 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/bin/mine_matrix /usr/local/bin/mine_matrix

WORKDIR /app
COPY mining/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY mining/mine_matrices.py .

# Pre-fetch the SuiteSparse index so workers don't race on first run
RUN python -c "import ssgetpy; ssgetpy.search(limit=1)" || true

VOLUME ["/data"]

ENTRYPOINT ["python", "mine_matrices.py"]
CMD ["--help"]
