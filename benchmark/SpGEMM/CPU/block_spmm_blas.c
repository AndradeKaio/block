#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_max_threads(void) { return 1; }
static int omp_get_thread_num(void) { return 0; }
#endif

#ifdef USE_FLOAT
typedef float scalar_t;
#define GEMM cblas_sgemm
static const float ONE = 1.0f;
#else
typedef double scalar_t;
#define GEMM cblas_dgemm
static const double ONE = 1.0;
#endif

void cblas_dgemm(int, int, int, int, int, int, double, const double *, int,
                 const double *, int, double, double *, int);
void cblas_sgemm(int, int, int, int, int, int, float, const float *, int,
                 const float *, int, float, float *, int);

void block_spmm(const scalar_t *A_data, const scalar_t *B_data,
                scalar_t *C_data, int C_size, int NC, int NG, const int *M_v,
                const int *N_v, const int *K_v, const long *A_off,
                const int *A_lda, const long *B_off, const int *B_ldb,
                const long *C_goff, const int *C_ldc, const int *G_start) {
  memset(C_data, 0, (size_t)C_size * sizeof(scalar_t));

  if (NG > 1) {
#pragma omp parallel for schedule(dynamic)
    for (int g = 0; g < NG; g++) {
      for (int ci = G_start[g]; ci < G_start[g + 1]; ci++) {
        GEMM(101, 111, 111, M_v[ci], N_v[ci], K_v[ci], ONE, A_data + A_off[ci],
             A_lda[ci], B_data + B_off[ci], B_ldb[ci], ONE, C_data + C_goff[ci],
             C_ldc[ci]);
      }
    }
  } else {
    long out_base = C_goff[0];
    int nt = omp_get_max_threads();
    scalar_t *tbufs = (scalar_t *)calloc((size_t)nt * C_size, sizeof(scalar_t));
#pragma omp parallel for schedule(dynamic)
    for (int ci = 0; ci < NC; ci++) {
      int tid = omp_get_thread_num();
      scalar_t *tb = tbufs + (size_t)tid * C_size;
      GEMM(101, 111, 111, M_v[ci], N_v[ci], K_v[ci], ONE, A_data + A_off[ci],
           A_lda[ci], B_data + B_off[ci], B_ldb[ci], ONE,
           tb + (C_goff[ci] - out_base), C_ldc[ci]);
    }
    scalar_t *C_out = C_data + out_base;
    for (int t = 0; t < nt; t++) {
      scalar_t *tb = tbufs + (size_t)t * C_size;
      for (long i = 0; i < C_size; i++)
        C_out[i] += tb[i];
    }
    free(tbufs);
  }
}
