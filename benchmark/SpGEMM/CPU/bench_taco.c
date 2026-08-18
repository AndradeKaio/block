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

int main(int argc, char *argv[]) {
  if (argc < 3) {
    fprintf(stderr, "usage: %s A.mtx B.mtx [n_runs]\n", argv[0]);
    return 1;
  }
  int n = argc > 3 ? atoi(argv[3]) : 1;

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

  /* Symbolic phase (assemble) runs once — determines output sparsity pattern.
     Timing it separately lets the timed loop measure compute in isolation,
     matching how Prisma amortises its symbolic pipeline. */
  long sym_t0 = ns_now();
  assemble(A_t, B_t, C_t);
  long sym_ns = ns_now() - sym_t0;
  int A_nnz = ((int *)A_t->indices[1][0])[M];
  printf("assemble_ns=%ld\n", sym_ns);
  printf("A_nnz=%d\n", A_nnz);
  fflush(stdout);

  /* Warmup compute (run_id 0), then n-1 timed runs.
     Zero vals before each call so repeated compute gives correct results. */
  long c_total = 0;
  for (int r = 0; r < n; r++) {
    memset(A_t->vals, 0, (size_t)A_nnz * sizeof(double));
    long t0 = ns_now();
    compute(A_t, B_t, C_t);
    long c = ns_now() - t0;
    c_total += c;
    printf("run_%d_compute_ns=%ld\n", r, c);
    fflush(stdout);
  }

  printf("mean_compute_ns=%ld\n", c_total / n);

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
