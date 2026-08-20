#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include TACO_KERNEL_H

typedef struct {
  int row, col;
  double val;
} entry_t;

static int entry_cmp(const void *a, const void *b) {
  const entry_t *ea = a, *eb = b;
  return ea->row != eb->row ? ea->row - eb->row : ea->col - eb->col;
}

static int read_mtx(const char *path, int *M, int *N, int *nnz, int **rows,
                    int **cols, double **vals) {
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    return -1;
  }
  char line[512];
  do {
    if (!fgets(line, sizeof(line), f)) {
      fclose(f);
      return -1;
    }
  } while (line[0] == '%');
  if (sscanf(line, "%d %d %d", M, N, nnz) != 3) {
    fclose(f);
    return -1;
  }
  entry_t *e = malloc(*nnz * sizeof(entry_t));
  for (int i = 0; i < *nnz; i++) {
    fscanf(f, "%d %d %lf", &e[i].row, &e[i].col, &e[i].val);
    e[i].row--;
    e[i].col--;
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

static taco_tensor_t *make_csr(int nrows, int ncols) {
  int32_t dims[2] = {nrows, ncols}, ord[2] = {0, 1};
  taco_mode_t modes[2] = {taco_mode_dense, taco_mode_sparse};
  taco_tensor_t *t = init_taco_tensor_t(2, sizeof(double), dims, ord, modes);
  t->indices[0][0] = t->indices[1][0] = t->indices[1][1] = t->vals = NULL;
  t->fill_value = NULL;
  t->vals_size = 0;
  return t;
}

static void reset_out(taco_tensor_t *A) {
  if (A->indices[1][0]) {
    free(A->indices[1][0]);
    A->indices[1][0] = NULL;
  }
  if (A->indices[1][1]) {
    free(A->indices[1][1]);
    A->indices[1][1] = NULL;
  }
  if (A->vals) {
    free(A->vals);
    A->vals = NULL;
  }
}

static long ns_now(void) {
  struct timespec ts;
  clock_gettime(CLOCK_MONOTONIC, &ts);
  return (long)ts.tv_sec * 1000000000L + ts.tv_nsec;
}

/* Dump the computed A (CSR: mode 0 dense/row, mode 1 sparse/col) as bare
   "row col val" COO text, 0-indexed -- same format prisma_cpu_bench.cpp's
   --validate writes, so validate_spgemm_cpu.py's existing _load_coo can
   read either contender's output identically. */
static void dump_csr_coo(const char *path, taco_tensor_t *A, int M) {
  FILE *f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "bench_taco: cannot open dump-c path: %s\n", path);
    return;
  }
  const int *row_ptr = (const int *)A->indices[1][0];
  const int *col_idx = (const int *)A->indices[1][1];
  const double *vals = (const double *)A->vals;
  for (int i = 0; i < M; i++) {
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; k++) {
      fprintf(f, "%d %d %.17g\n", i, col_idx[k], vals[k]);
    }
  }
  fclose(f);
}

int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s A.mtx B.mtx [n_runs] [--dump-c path]\n", argv[0]);
    return 1;
  }
  int n = argc > 3 && argv[3][0] != '-' ? atoi(argv[3]) : 1;
  const char *dump_c_path = NULL;
  for (int i = 3; i < argc; i++) {
    if (strcmp(argv[i], "--dump-c") == 0 && i + 1 < argc) {
      dump_c_path = argv[++i];
    }
  }

  int M, K, K2, N, B_nnz, C_nnz;
  int *Br, *Bc, *Cr, *Cc;
  double *Bv, *Cv;

  if (read_mtx(argv[1], &M, &K, &B_nnz, &Br, &Bc, &Bv) != 0)
    return 1;
  if (read_mtx(argv[2], &K2, &N, &C_nnz, &Cr, &Cc, &Cv) != 0)
    return 1;
  if (K != K2) {
    fprintf(stderr, "dimension mismatch\n");
    return 1;
  }

  taco_tensor_t *B_t = make_csr(M, K);
  taco_tensor_t *C_t = make_csr(K, N);
  taco_tensor_t *A_t = make_csr(M, N);

  int Bp[2] = {0, B_nnz}, Cp[2] = {0, C_nnz};
  pack_B(B_t, Bp, Br, Bc, Bv);
  pack_C(C_t, Cp, Cr, Cc, Cv);

  /* Every run redoes assemble() from scratch (freeing and reallocating A_t's
     output arrays first via reset_out) instead of assembling once and
     reusing the pattern -- matches prisma_cpu_bench.cpp's equivalent change.
     A caller who invokes SpGEMM once pays the full assemble() cost, not an
     artificially amortised fraction of it; averaging N real measurements
     (which benchmark_spgemm_cpu.py's plot does via a plain mean) reflects
     that honestly. */
  long sym_total = 0, c_total = 0;
  int A_nnz = 0;
  for (int r = 0; r < n; r++) {
    if (r > 0)
      reset_out(A_t);
    long t0 = ns_now();
    assemble(A_t, B_t, C_t);
    long sym = ns_now() - t0;
    sym_total += sym;
    A_nnz = ((int *)A_t->indices[1][0])[M];
    memset(A_t->vals, 0, (size_t)A_nnz * sizeof(double));
    long t1 = ns_now();
    compute(A_t, B_t, C_t);
    long c = ns_now() - t1;
    c_total += c;
    printf("run_%d_assemble_ns=%ld\n", r, sym);
    printf("run_%d_compute_ns=%ld\n", r, c);
    fflush(stdout);
  }

  printf("A_nnz=%d\n", A_nnz);
  printf("mean_assemble_ns=%ld\n", sym_total / n);
  printf("mean_compute_ns=%ld\n", c_total / n);

  if (dump_c_path)
    dump_csr_coo(dump_c_path, A_t, M);

  free(Br);
  free(Bc);
  free(Bv);
  free(Cr);
  free(Cc);
  free(Cv);
  reset_out(A_t);
  deinit_taco_tensor_t(A_t);
  deinit_taco_tensor_t(B_t);
  deinit_taco_tensor_t(C_t);
  return 0;
}
