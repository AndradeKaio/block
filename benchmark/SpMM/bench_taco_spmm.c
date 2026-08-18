// bench_taco_spmm.c — SpMM benchmark: D = S * D using TACO-generated kernel
//
// Loads a single sparse matrix S from MTX; the dense operand is constructed
// in-memory as a K×K all-ones matrix (K = number of columns of S, always
// square).  Timing is split into symbolic (assemble) and numeric (compute)
// phases, matching the format of bench_taco.c so the same Python parser
// applies.
//
// CLI: bench_taco_spmm <S.mtx> [--runs R]
//   R  number of timed repetitions (default 5); run 0 is warmup
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
//   gcc -O3 -fopenmp -I. bench_taco_spmm.c -lm -o bench_taco_spmm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#if defined(KERNEL_OPT_0)
#  include "taco_kernel_opt_0.h"
#elif defined(KERNEL_OPT_1)
#  include "taco_kernel_opt_1.h"
#else
#  include "taco_kernel.h"
#endif

typedef struct { int row, col; double val; } entry_t;

static int entry_cmp(const void *a, const void *b) {
    const entry_t *ea = a, *eb = b;
    return ea->row != eb->row ? ea->row - eb->row : ea->col - eb->col;
}

static int read_mtx(const char *path, int *M, int *N, int *nnz,
                    int **rows, int **cols, double **vals) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    char line[512];
    // First line: %%MatrixMarket header — check for "pattern"
    int is_pattern = 0;
    if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }
    if (strstr(line, "pattern") || strstr(line, "Pattern")) is_pattern = 1;
    // Skip remaining comment lines
    do {
        if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }
    } while (line[0] == '%');
    if (sscanf(line, "%d %d %d", M, N, nnz) != 3) { fclose(f); return -1; }
    entry_t *e = malloc(*nnz * sizeof(entry_t));
    for (int i = 0; i < *nnz; i++) {
        if (is_pattern) {
            fscanf(f, "%d %d", &e[i].row, &e[i].col);
            e[i].val = 1.0;
        } else {
            fscanf(f, "%d %d %lf", &e[i].row, &e[i].col, &e[i].val);
        }
        e[i].row--; e[i].col--;
    }
    fclose(f);
    qsort(e, *nnz, sizeof(entry_t), entry_cmp);
    *rows = malloc(*nnz * sizeof(int));
    *cols = malloc(*nnz * sizeof(int));
    *vals = malloc(*nnz * sizeof(double));
    for (int i = 0; i < *nnz; i++) {
        (*rows)[i] = e[i].row;
        (*cols)[i] = e[i].col;
        (*vals)[i] = e[i].val;
    }
    free(e);
    return 0;
}

// Allocate a sparse (dense-sparse / CSR) taco tensor — used for B only.
static taco_tensor_t *make_sparse(int nrows, int ncols) {
    int32_t dims[2] = {nrows, ncols}, ord[2] = {0, 1};
    taco_mode_t modes[2] = {taco_mode_dense, taco_mode_sparse};
    taco_tensor_t *t = init_taco_tensor_t(2, sizeof(double), dims, ord, modes);
    t->indices[0][0] = t->indices[1][0] = t->indices[1][1] = t->vals = NULL;
    t->fill_value = NULL;
    t->vals_size = 0;
    return t;
}

// Allocate a dense (dense-dense) taco tensor — used for A and C.
static taco_tensor_t *make_dense(int nrows, int ncols) {
    int32_t dims[2] = {nrows, ncols}, ord[2] = {0, 1};
    taco_mode_t modes[2] = {taco_mode_dense, taco_mode_dense};
    taco_tensor_t *t = init_taco_tensor_t(2, sizeof(double), dims, ord, modes);
    t->vals = NULL;
    t->fill_value = NULL;
    t->vals_size = 0;
    return t;
}

// Free output A's vals so assemble() can reallocate cleanly (A is dense-dense).
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
        fprintf(stderr, "usage: %s S.mtx [--runs R]\n", argv[0]);
        return 1;
    }
    const char *mtx_path = argv[1];
    int n_runs = 5;
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--runs") == 0 && i + 1 < argc)
            n_runs = atoi(argv[++i]);
    }

    int M, N, S_nnz;
    int *Sr, *Sc;
    double *Sv;
    if (read_mtx(mtx_path, &M, &N, &S_nnz, &Sr, &Sc, &Sv) != 0) return 1;

    int K = N; // matrix is always square; num_vecs = K

    // Build sparse B (= S) by packing COO into CSR via pack_B
    taco_tensor_t *B_t = make_sparse(M, K);
    int Sp[2] = {0, S_nnz};
    pack_B(B_t, Sp, Sr, Sc, Sv);

    // Build dense C (K × K, random in [-1,1]) — bypass pack_C, set vals directly
    taco_tensor_t *C_t = make_dense(K, K);
    double *c_vals = malloc((size_t)K * K * sizeof(double));
    srand(42);
    for (int i = 0; i < K * K; i++)
        c_vals[i] = (double)rand() / RAND_MAX * 2.0 - 1.0;
    C_t->vals = (uint8_t *)c_vals;

    // Output A — dense-dense (both kernels)
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

    printf("mean_assemble_ns=%ld\n", a_total / n_runs);
    printf("mean_compute_ns=%ld\n",  c_total / n_runs);
    printf("S_nnz=%d\n",    S_nnz);
    printf("S_rows=%d\n",   M);
    printf("S_cols=%d\n",   N);
    printf("num_vecs=%d\n", K);
    printf("A_nnz=%d\n",    A_nnz);

    free(Sr); free(Sc); free(Sv);
    free(c_vals);
    reset_output(A_t);
    deinit_taco_tensor_t(A_t);
    // B_t's arrays are owned by pack_B
    if (B_t->indices[1][0]) free(B_t->indices[1][0]);
    if (B_t->indices[1][1]) free(B_t->indices[1][1]);
    if (B_t->vals)          free(B_t->vals);
    deinit_taco_tensor_t(B_t);
    deinit_taco_tensor_t(C_t);
    return 0;
}
