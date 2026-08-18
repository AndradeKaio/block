// bench_tc_spgemm.cu — standalone harness for TC_SpGEMM with per-phase timing
//
// Usage: bench_tc_spgemm A.mtx B.mtx [--output C.mtx]
//
// Prints three separate timings:
//   TC SpGEMM preprocessing time is X.XXXX ms
//   TC SpGEMM compute time is X.XXXX ms
//   TC SpGEMM postprocess time is X.XXXX ms
//
// Compile:
//   nvcc -O3 -arch=sm_120 -std=c++17 -DTC_SPGEMM_NO_MAIN \
//        bench_tc_spgemm.cu -o bench_tc_spgemm

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define TC_SPGEMM_NO_MAIN
#include "tc_spgemm.cu"

// ---------------------------------------------------------------------------
// .mtx reader — returns 0-indexed COO triplets, sorted row-then-col.
// Identical pattern to bench_taco_gpu.cu.
// ---------------------------------------------------------------------------

typedef struct { int row, col; float val; } entry_t;

static int entry_cmp(const void *a, const void *b) {
    const entry_t *ea = (const entry_t *)a, *eb = (const entry_t *)b;
    return ea->row != eb->row ? ea->row - eb->row : ea->col - eb->col;
}

static int read_mtx(const char *path, int *M, int *N, int *nnz,
                    int **rows, int **cols, float **vals) {
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    char line[512];
    do {
        if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; }
    } while (line[0] == '%');
    if (sscanf(line, "%d %d %d", M, N, nnz) != 3) { fclose(f); return -1; }
    entry_t *e = (entry_t *)malloc(*nnz * sizeof(entry_t));
    for (int i = 0; i < *nnz; i++) {
        float v = 1.0f;
        int matched = fscanf(f, "%d %d %f", &e[i].row, &e[i].col, &v);
        if (matched < 2) { free(e); fclose(f); return -1; }
        e[i].val = (matched == 3) ? v : 1.0f;
        e[i].row--; e[i].col--;
    }
    fclose(f);
    qsort(e, *nnz, sizeof(entry_t), entry_cmp);
    *rows = (int *)malloc(*nnz * sizeof(int));
    *cols = (int *)malloc(*nnz * sizeof(int));
    *vals = (float *)malloc(*nnz * sizeof(float));
    for (int i = 0; i < *nnz; i++) {
        (*rows)[i] = e[i].row;
        (*cols)[i] = e[i].col;
        (*vals)[i] = e[i].val;
    }
    free(e);
    return 0;
}

static void coo_to_csr(const int *rows, int nnz, int M, int *rowptr) {
    memset(rowptr, 0, (M + 1) * sizeof(int));
    for (int i = 0; i < nnz; i++) rowptr[rows[i] + 1]++;
    for (int i = 0; i < M; i++) rowptr[i + 1] += rowptr[i];
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static double elapsed_ms(const struct timeval *t1, const struct timeval *t2) {
    return (t2->tv_sec  - t1->tv_sec)  * 1000.0
         + (t2->tv_usec - t1->tv_usec) / 1000.0;
}

static void write_coo_mtx(const char *path, int M, int N,
                           const int *row, const int *col, const float *val, int nnz) {
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return; }
    fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(f, "%d %d %d\n", M, N, nnz);
    for (int i = 0; i < nnz; i++)
        fprintf(f, "%d %d %.9g\n", row[i]+1, col[i]+1, val[i]);
    fclose(f);
    printf("Wrote C -> %s\n", path);
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s A.mtx B.mtx [--output C.mtx]\n", argv[0]);
        return 1;
    }

    const char *output_path = NULL;
    int runs = 1;
    for (int i = 3; i < argc - 1; i++) {
        if (strcmp(argv[i], "--output") == 0) { output_path = argv[i + 1]; }
        if (strcmp(argv[i], "--runs")   == 0) { runs = atoi(argv[i + 1]); }
    }

    cudaSetDevice(0);

    // ── Read A and B from disk (not timed) ───────────────────────────────
    int M, K, K2, N, A_nnz, B_nnz;
    int *Ar, *Ac, *Br, *Bc;
    float *Av, *Bv;

    if (read_mtx(argv[1], &M, &K,  &A_nnz, &Ar, &Ac, &Av) != 0) return 1;
    if (read_mtx(argv[2], &K2, &N, &B_nnz, &Br, &Bc, &Bv) != 0) return 1;
    if (K != K2) {
        fprintf(stderr, "dimension mismatch: A cols=%d vs B rows=%d\n", K, K2);
        return 1;
    }

    int *A_rowptr = (int *)malloc((M + 1) * sizeof(int));
    int *B_rowptr = (int *)malloc((K + 1) * sizeof(int));
    coo_to_csr(Ar, A_nnz, M, A_rowptr);
    coo_to_csr(Br, B_nnz, K, B_rowptr);

    printf("A: %d x %d  nnz=%d\n", M, K, A_nnz);
    printf("B: %d x %d  nnz=%d\n", K, N, B_nnz);

    // ── Preprocessing (done once, outside timing loop) ────────────────────
    CSRDense hA{}, hB{};
    preprocess_A(M, K, A_rowptr, Ac, Av, hA);
    preprocess_B(K, N, B_rowptr, Bc, Bv, hB);

    // ── Warmup (run_id=0) + timed runs (run_id=1..runs) ──────────────────
    // symbolic = pair generation; compute = TC kernel + postprocess; total = wall clock.
    double *tc_sym_ms  = (double *)malloc((runs + 1) * sizeof(double));
    double *tc_cmp_ms  = (double *)malloc((runs + 1) * sizeof(double));
    double *tc_spgemm_ms = (double *)malloc((runs + 1) * sizeof(double));

    int   *C_row = NULL, *C_col = NULL;
    float *C_val = NULL;
    int    C_nnz = 0;

    struct timeval t1, t2;
    for (int r = 0; r <= runs; r++) {
        RawCOO raw{};
        gettimeofday(&t1, NULL);
        ComputeTimes ct = tc_spgemm_compute(hA, hB, N, raw);
        int *row_r = NULL, *col_r = NULL; float *val_r = NULL; int nnz_r = 0;
        cudaEvent_t _ev0, _ev1;
        cudaEventCreate(&_ev0); cudaEventCreate(&_ev1);
        cudaEventRecord(_ev0);
        tc_spgemm_postprocess(raw, N, &row_r, &col_r, &val_r, &nnz_r);
        cudaEventRecord(_ev1);
        cudaEventSynchronize(_ev1);
        float _post_ms_f; cudaEventElapsedTime(&_post_ms_f, _ev0, _ev1);
        cudaEventDestroy(_ev0); cudaEventDestroy(_ev1);
        gettimeofday(&t2, NULL);

        tc_sym_ms[r]    = ct.symbolic_ms;
        tc_cmp_ms[r]    = ct.kernel_ms + (double)_post_ms_f;
        tc_spgemm_ms[r] = elapsed_ms(&t1, &t2);

        if (r < runs) {
            free(row_r); free(col_r); free(val_r);
        } else {
            C_row = row_r; C_col = col_r; C_val = val_r; C_nnz = nnz_r;
        }
    }

    printf("\nJSON_BEGIN\n{\n");
    printf("  \"tc_spgemm_symbolic_ms\": [");
    for (int r = 0; r <= runs; r++) { if (r) printf(", "); printf("%.4f", tc_sym_ms[r]); }
    printf("],\n  \"tc_spgemm_compute_ms\": [");
    for (int r = 0; r <= runs; r++) { if (r) printf(", "); printf("%.4f", tc_cmp_ms[r]); }
    printf("],\n  \"tc_spgemm_ms\": [");
    for (int r = 0; r <= runs; r++) { if (r) printf(", "); printf("%.4f", tc_spgemm_ms[r]); }
    printf("]\n}\nJSON_END\n");
    free(tc_sym_ms);
    free(tc_cmp_ms);
    free(tc_spgemm_ms);

    // ── Optional output ───────────────────────────────────────────────────
    if (output_path)
        write_coo_mtx(output_path, M, N, C_row, C_col, C_val, C_nnz);

    // ── Cleanup ───────────────────────────────────────────────────────────
    free_csrdense(hA);
    free_csrdense(hB);
    if (C_row) { free(C_row); free(C_col); free(C_val); }
    free(Ar); free(Ac); free(Av);
    free(Br); free(Bc); free(Bv);
    free(A_rowptr); free(B_rowptr);

    return 0;
}
