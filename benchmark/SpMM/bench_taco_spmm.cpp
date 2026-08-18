// bench_taco_spmm.cpp — SpMM benchmark: A = S * C using TACO-generated kernel
//
// Loads a single sparse matrix S from MTX; the dense operand is constructed
// in-memory as a K×K random matrix.  Timing is split into symbolic (assemble)
// and numeric (compute) phases, matching the format of bench_taco.c so the
// same Python parser applies.
//
// CLI: bench_taco_spmm <S.mtx> [--runs R] [--seed S] [--dump-c path]
//   R     number of timed repetitions (default 5); run 0 is warmup
//   S     RNG seed for the dense input (default 42)
//   path  write the M×K output A as raw doubles to this file after the last run
//
// Output:
//   run_N_assemble_ns=<long>
//   run_N_compute_ns=<long>
//   ...
//   mean_assemble_ns=<long>   (runs 1..R, warmup excluded)
//   mean_compute_ns=<long>
//   S_nnz=<int>
//   S_rows=<int>
//   S_cols=<int>
//   num_vecs=<int>            (equals S_cols = S_rows)
//
// Compile:
//   g++ -O3 -std=c++17 -fopenmp -march=native -Drestrict=__restrict__ -I. \
//       bench_taco_spmm.cpp -lm -o bench_taco_spmm
//   (-march=native matters: without it GCC only auto-vectorizes compute()'s
//   loop to baseline SSE2, a big handicap vs AVX-512-compiled contenders.)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <random>
#include <string>

#if defined(KERNEL_OPT_0)
#  include "taco_kernel_opt_0.h"
#elif defined(KERNEL_OPT_1)
#  include "taco_kernel_opt_1.h"
#else
#  include "taco_kernel.h"
#endif

typedef struct { int row, col; double val; } entry_t;

static int entry_cmp(const void *a, const void *b) {
    const entry_t *ea = (const entry_t *)a, *eb = (const entry_t *)b;
    return ea->row != eb->row ? ea->row - eb->row : ea->col - eb->col;
}

static int read_mtx(const char *path, int *M, int *N, int *nnz,
                    int **rows, int **cols, double **vals) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    char line[512];

    // The MatrixMarket format/field/symmetry flags live ONLY in the very
    // first line (the "%%MatrixMarket matrix coordinate <field> <symmetry>"
    // banner) — that's fixed by the spec. Free-text comment lines that
    // follow are documentation, not part of the format, and must not be
    // scanned for these keywords: real-world files routinely use the
    // English word "pattern" in prose (e.g. "the sparsity pattern of the
    // matrix..."), which previously false-positived is_pattern and silently
    // corrupted the parse of real-valued matrices (verified on
    // Cylshell/s1rmt3m1.mtx: read 1 of 112505 entries).
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }
    char banner[512];
    std::strncpy(banner, line, sizeof(banner));
    banner[sizeof(banner) - 1] = '\0';
    for (char *p = banner; *p; ++p) *p = (char)tolower((unsigned char)*p);
    int is_pattern   = strstr(banner, "pattern") != nullptr;
    int is_symmetric = strstr(banner, "symmetric") || strstr(banner, "hermitian");
    if (strstr(banner, "complex")) {
        // Complex-valued matrices aren't supported by this real-only SpMM
        // benchmark (or by the prisma/TACO kernels it compares against) —
        // fail loudly instead of silently misreading every other value as
        // real/imaginary pairs, which previously corrupted every entry
        // (verified on QCD/conf5_0-4x4-*.mtx).
        fprintf(stderr, "read_mtx: complex-valued matrices are not supported: %s\n", path);
        fclose(f);
        return -1;
    }

    // Skip the remaining %-comment lines without inspecting their content.
    while (fgets(line, sizeof(line), f)) {
        if (line[0] != '%') break;
    }
    // `line` now holds the first non-comment line: M N NNZ
    if (sscanf(line, "%d %d %d", M, N, nnz) != 3) { fclose(f); return -1; }

    // Reserve up to 2× for symmetric matrices (off-diagonal entries are mirrored)
    int cap = is_symmetric ? *nnz * 2 : *nnz;
    entry_t *e = (entry_t *)malloc((size_t)cap * sizeof(entry_t));
    int n = 0;
    for (int i = 0; i < *nnz; i++) {
        int r, c; double v = 1.0;
        if (is_pattern) {
            if (fscanf(f, "%d %d",      &r, &c)     < 2) continue;
        } else {
            if (fscanf(f, "%d %d %lf", &r, &c, &v) < 2) continue;
        }
        --r; --c;
        e[n++] = {r, c, v};
        if (is_symmetric && r != c)
            e[n++] = {c, r, v};
    }
    fclose(f);
    qsort(e, n, sizeof(entry_t), entry_cmp);
    *nnz  = n;
    *rows = (int *)   malloc((size_t)n * sizeof(int));
    *cols = (int *)   malloc((size_t)n * sizeof(int));
    *vals = (double *)malloc((size_t)n * sizeof(double));
    for (int i = 0; i < n; i++) {
        (*rows)[i] = e[i].row;
        (*cols)[i] = e[i].col;
        (*vals)[i] = e[i].val;
    }
    free(e);
    return 0;
}

static taco_tensor_t *make_sparse(int nrows, int ncols) {
    int32_t dims[2] = {nrows, ncols}, ord[2] = {0, 1};
    taco_mode_t modes[2] = {taco_mode_dense, taco_mode_sparse};
    taco_tensor_t *t = init_taco_tensor_t(2, sizeof(double), dims, ord, modes);
    t->indices[0][0] = t->indices[1][0] = t->indices[1][1] = t->vals = NULL;
    t->fill_value = NULL;
    t->vals_size = 0;
    return t;
}

static taco_tensor_t *make_dense(int nrows, int ncols) {
    int32_t dims[2] = {nrows, ncols}, ord[2] = {0, 1};
    taco_mode_t modes[2] = {taco_mode_dense, taco_mode_dense};
    taco_tensor_t *t = init_taco_tensor_t(2, sizeof(double), dims, ord, modes);
    t->vals = NULL;
    t->fill_value = NULL;
    t->vals_size = 0;
    return t;
}

static void reset_output(taco_tensor_t *A) {
    if (A->vals) { free(A->vals); A->vals = NULL; }
}

static long ns_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s S.mtx [--runs R] [--seed S] [--dump-c path]\n", argv[0]);
        return 1;
    }
    const char *mtx_path = argv[1];
    int n_runs = 5;
    int seed   = 42;
    const char *dump_c_path = nullptr;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc)
            n_runs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc)
            seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--dump-c") == 0 && i + 1 < argc)
            dump_c_path = argv[++i];
    }

    int M, N, S_nnz;
    int *Sr, *Sc;
    double *Sv;
    long t_read0 = ns_now();
    if (read_mtx(mtx_path, &M, &N, &S_nnz, &Sr, &Sc, &Sv) != 0) return 1;
    long mtx_read_ns = ns_now() - t_read0;

    int K = N;

    // pack_B is TACO's one-time structural setup (builds B's CSR index
    // structure from the COO triples read above), done once before the
    // run loop, same category of cost as prisma_cpu_spmm_bench's row-group
    // build — see that file's matching instrumentation, added at the same
    // time so the two numbers are directly comparable instead of both
    // being silently excluded from every reported number as before.
    taco_tensor_t *B_t = make_sparse(M, K);
    int Sp[2] = {0, S_nnz};
    long t_pack0 = ns_now();
    pack_B(B_t, Sp, Sr, Sc, Sv);
    long pack_b_ns = ns_now() - t_pack0;
    fprintf(stderr, "structural_setup: mtx_read=%.4fms  pack_b=%.4fms\n",
           mtx_read_ns / 1e6, pack_b_ns / 1e6);

    // Dense input C (K×K), same RNG as prisma_cpu_spmm_bench: mt19937_64
    taco_tensor_t *C_t = make_dense(K, K);
    double *c_vals = (double *)malloc((size_t)K * K * sizeof(double));
    {
        std::mt19937_64 rng((uint64_t)seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (long long i = 0; i < (long long)K * K; i++)
            c_vals[i] = dist(rng);
    }
    C_t->vals = (uint8_t *)c_vals;

    taco_tensor_t *A_t = make_dense(M, K);

    long a_total = 0, c_total = 0;
    int A_nnz = 0;

    for (int r = 0; r <= n_runs; r++) {
        reset_output(A_t);
        long t0 = ns_now();
        assemble(A_t, B_t, C_t);
        long t1 = ns_now();
        compute(A_t, B_t, C_t);
        long t2 = ns_now();
        long a = t1 - t0, c = t2 - t1;
        if (r > 0) { a_total += a; c_total += c; }
        if (r == 0) A_nnz = M * K;
        printf("run_%d_assemble_ns=%ld\n", r, a);
        printf("run_%d_compute_ns=%ld\n",  r, c);
        fflush(stdout);
    }

    // Dump output A (M×K row-major doubles) if requested
    if (dump_c_path) {
        FILE *fout = fopen(dump_c_path, "wb");
        if (!fout) {
            fprintf(stderr, "bench_taco_spmm: cannot open dump-c path: %s\n", dump_c_path);
        } else {
            fwrite((double *)A_t->vals, sizeof(double), (size_t)M * K, fout);
            fclose(fout);
        }
    }

    printf("mean_assemble_ns=%ld\n", a_total / n_runs);
    printf("mean_compute_ns=%ld\n",  c_total / n_runs);
    printf("mtx_read_ns=%ld\n", mtx_read_ns);
    printf("pack_b_ns=%ld\n",   pack_b_ns);
    printf("S_nnz=%d\n",    S_nnz);
    printf("S_rows=%d\n",   M);
    printf("S_cols=%d\n",   N);
    printf("num_vecs=%d\n", K);
    printf("A_nnz=%d\n",    A_nnz);

    free(Sr); free(Sc); free(Sv);
    free(c_vals);
    reset_output(A_t);
    deinit_taco_tensor_t(A_t);
    if (B_t->indices[1][0]) free(B_t->indices[1][0]);
    if (B_t->indices[1][1]) free(B_t->indices[1][1]);
    if (B_t->vals)          free(B_t->vals);
    deinit_taco_tensor_t(B_t);
    deinit_taco_tensor_t(C_t);
    return 0;
}
