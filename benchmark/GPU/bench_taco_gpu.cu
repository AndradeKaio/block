// bench_taco_gpu.cu — CUDA harness for TACO-generated SpGEMM kernel
// Usage: bench_taco_gpu A.mtx B.mtx
//
// Reads two CSR matrices, runs one warmup, then one timed call to
// compute(C, A, B) from core/taco_spgemm.cu. C is a dense M×N output.
// Prints: "TACO GPU runtime is X.XXXX ms"
//
// NOTE: compute() zero-initialises the full M×N float buffer on the
// host before each GPU kernel launch. This cost is included in the
// reported time and scales as O(M×N).

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <cuda_runtime.h>

// --- Stubs for TACO COO pack/unpack globals ------------------------------------
// pack_A / pack_B / unpack in taco_spgemm.cu reference these names, but those
// functions are never called. Declaring stubs here satisfies the compiler.
static int    *A_COO1_pos = nullptr, *A_COO1_crd = nullptr, *A_COO2_crd = nullptr;
static float  *A_COO_vals  = nullptr;
static int    *B_COO1_pos = nullptr, *B_COO1_crd = nullptr, *B_COO2_crd = nullptr;
static float  *B_COO_vals  = nullptr;
static int    **C_COO1_pos_ptr = nullptr, **C_COO1_crd_ptr = nullptr,
              **C_COO2_crd_ptr = nullptr;
static float  **C_COO_vals_ptr = nullptr;

// --- TACO generated kernel (dense C, CSR A and B) ----------------------------
#include "taco_spgemm.cu"

// ---------------------------------------------------------------------------
// Minimal .mtx reader (same logic as bench_taco.c)
// Returns 0-indexed COO triplets, sorted row-then-col.
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
        e[i].val = (matched == 3) ? v : 1.0;
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

// ---------------------------------------------------------------------------
// COO → CSR rowptr on host
// ---------------------------------------------------------------------------

static void coo_to_csr(const int *rows, int nnz, int M, int *rowptr) {
    memset(rowptr, 0, (M + 1) * sizeof(int));
    for (int i = 0; i < nnz; i++) rowptr[rows[i] + 1]++;
    for (int i = 0; i < M; i++) rowptr[i + 1] += rowptr[i];
}

// ---------------------------------------------------------------------------
// Build a taco_tensor_t for a CSR matrix in cudaMallocManaged memory.
//
// All sub-pointers must be device-accessible because computeDeviceKernel0
// receives taco_tensor_t* and dereferences members on the device.
// ---------------------------------------------------------------------------

static taco_tensor_t *make_csr_managed(int nrows, int ncols,
                                       const int *h_rowptr,
                                       const int *h_colidx,
                                       const float *h_vals, int nnz) {
    taco_tensor_t *t;
    cudaMallocManaged(&t, sizeof(taco_tensor_t));
    memset(t, 0, sizeof(taco_tensor_t));

    t->order = 2;
    t->csize = (int32_t)sizeof(float);
    t->fill_value = nullptr;
    t->mode_ordering = nullptr; // not accessed by the kernel
    t->mode_types    = nullptr; // not accessed by the kernel

    int32_t *dims;
    cudaMallocManaged(&dims, 2 * sizeof(int32_t));
    dims[0] = nrows; dims[1] = ncols;
    t->dimensions = dims;

    // indices[0] = NULL (dense first mode, never dereferenced by kernel)
    // indices[1] = { rowptr, colidx }
    uint8_t ***indices;
    cudaMallocManaged(&indices, 2 * sizeof(uint8_t **));
    indices[0] = nullptr;

    uint8_t **mode1;
    cudaMallocManaged(&mode1, 2 * sizeof(uint8_t *));

    int32_t *d_rowptr;
    cudaMallocManaged(&d_rowptr, (nrows + 1) * sizeof(int32_t));
    memcpy(d_rowptr, h_rowptr, (nrows + 1) * sizeof(int32_t));

    int32_t *d_colidx;
    cudaMallocManaged(&d_colidx, nnz * sizeof(int32_t));
    memcpy(d_colidx, h_colidx, nnz * sizeof(int32_t));

    float *d_vals;
    cudaMallocManaged(&d_vals, nnz * sizeof(float));
    memcpy(d_vals, h_vals, nnz * sizeof(float));

    mode1[0] = (uint8_t *)d_rowptr;
    mode1[1] = (uint8_t *)d_colidx;
    indices[1] = mode1;

    t->indices  = indices;
    t->vals     = (uint8_t *)d_vals;
    t->vals_size = nnz;

    return t;
}

// ---------------------------------------------------------------------------
// Build a taco_tensor_t for the dense M×N output C.
// Only dimensions needs to be managed; assemble() will allocate vals.
// ---------------------------------------------------------------------------

static taco_tensor_t *make_dense_managed(int nrows, int ncols) {
    taco_tensor_t *t;
    cudaMallocManaged(&t, sizeof(taco_tensor_t));
    memset(t, 0, sizeof(taco_tensor_t));

    t->order = 2;
    t->csize = (int32_t)sizeof(float);

    int32_t *dims;
    cudaMallocManaged(&dims, 2 * sizeof(int32_t));
    dims[0] = nrows; dims[1] = ncols;
    t->dimensions = dims;

    // assemble() allocates t->vals via cudaMallocManaged
    // indices not needed for dense output
    t->vals    = nullptr;
    t->indices = nullptr;

    return t;
}

// ---------------------------------------------------------------------------

static double elapsed_ms(const struct timeval *t1, const struct timeval *t2) {
    return (t2->tv_sec  - t1->tv_sec)  * 1000.0
         + (t2->tv_usec - t1->tv_usec) / 1000.0;
}

static void write_dense_mtx(const char *path, taco_tensor_t *C) {
    int M = C->dimensions[0], N = C->dimensions[1];
    float *vals = (float *)C->vals;
    int nnz = 0;
    for (int i = 0; i < M * N; i++) if (vals[i] != 0.0f) nnz++;
    FILE *f = fopen(path, "w");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return; }
    fprintf(f, "%%%%MatrixMarket matrix coordinate real general\n");
    fprintf(f, "%d %d %d\n", M, N, nnz);
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float v = vals[i * N + j];
            if (v != 0.0f)
                fprintf(f, "%d %d %.9g\n", i+1, j+1, v);
        }
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

    // Read matrices from .mtx files
    int M, K, K2, N, A_nnz, B_nnz;
    int *Ar, *Ac, *Br, *Bc;
    float *Av, *Bv;

    if (read_mtx(argv[1], &M, &K,  &A_nnz, &Ar, &Ac, &Av) != 0) return 1;
    if (read_mtx(argv[2], &K2, &N, &B_nnz, &Br, &Bc, &Bv) != 0) return 1;
    if (K != K2) {
        fprintf(stderr, "dimension mismatch: A cols=%d vs B rows=%d\n", K, K2);
        return 1;
    }

    // Build host CSR rowptr arrays from sorted COO
    int *A_rowptr = (int *)malloc((M + 1) * sizeof(int));
    int *B_rowptr = (int *)malloc((K + 1) * sizeof(int));
    coo_to_csr(Ar, A_nnz, M, A_rowptr);
    coo_to_csr(Br, B_nnz, K, B_rowptr);

    // Build managed-memory taco_tensor_t structs
    taco_tensor_t *A_t = make_csr_managed(M, K, A_rowptr, Ac, Av, A_nnz);
    taco_tensor_t *B_t = make_csr_managed(K, N, B_rowptr, Bc, Bv, B_nnz);
    taco_tensor_t *C_t = make_dense_managed(M, N);

    // assemble: allocates dense M×N C->vals via cudaMallocManaged
    assemble(C_t, A_t, B_t);

    // Warmup (run_id=0) + timed runs (run_id=1..runs) — all recorded.
    // Timing is split into two phases:
    //   symbolic: cudaMemset zeroing the dense M×N C buffer (workspace setup)
    //   compute:  computeDeviceKernel0 GPU kernel only
    size_t C_bytes = (size_t)M * N * sizeof(float);
    double *taco_sym_ms  = (double *)malloc((runs + 1) * sizeof(double));
    double *taco_comp_ms = (double *)malloc((runs + 1) * sizeof(double));

    for (int r = 0; r <= runs; r++) {
        cudaEvent_t ev0, ev1, ev2;
        cudaEventCreate(&ev0);
        cudaEventCreate(&ev1);
        cudaEventCreate(&ev2);

        // Phase 1: zero-init C (symbolic proxy — scales as O(M×N))
        cudaEventRecord(ev0);
        cudaMemsetAsync(C_t->vals, 0, C_bytes);
        cudaEventRecord(ev1);

        // Phase 2: GPU kernel only
        computeDeviceKernel0<<<(M + 31) / 32, 32>>>(A_t, B_t, C_t);
        cudaEventRecord(ev2);
        cudaEventSynchronize(ev2);

        float ms1, ms2;
        cudaEventElapsedTime(&ms1, ev0, ev1);
        cudaEventElapsedTime(&ms2, ev1, ev2);
        cudaEventDestroy(ev0);
        cudaEventDestroy(ev1);
        cudaEventDestroy(ev2);

        taco_sym_ms[r]  = (double)ms1;
        taco_comp_ms[r] = (double)ms2;
    }

    printf("\nJSON_BEGIN\n{\n  \"taco_symbolic_ms\": [");
    for (int r = 0; r <= runs; r++) {
        if (r) printf(", ");
        printf("%.4f", taco_sym_ms[r]);
    }
    printf("],\n  \"taco_compute_ms\": [");
    for (int r = 0; r <= runs; r++) {
        if (r) printf(", ");
        printf("%.4f", taco_comp_ms[r]);
    }
    printf("],\n  \"taco_ms\": [");
    for (int r = 0; r <= runs; r++) {
        if (r) printf(", ");
        printf("%.4f", taco_sym_ms[r] + taco_comp_ms[r]);
    }
    printf("]\n}\nJSON_END\n");
    free(taco_sym_ms);
    free(taco_comp_ms);

    if (output_path)
        write_dense_mtx(output_path, C_t);

    free(Ar); free(Ac); free(Av);
    free(Br); free(Bc); free(Bv);
    free(A_rowptr); free(B_rowptr);
    // managed memory freed implicitly on process exit

    return 0;
}
