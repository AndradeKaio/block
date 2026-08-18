#include <stdlib.h>
#include <string.h>
#ifdef _OPENMP
#include <omp.h>
#else
static int omp_get_max_threads(void) { return 1; }
static int omp_get_thread_num(void)  { return 0; }
#endif

#ifdef USE_FLOAT
typedef float  scalar_t;
#else
typedef double scalar_t;
#endif

void block_spmm(
    const scalar_t* A_data, const scalar_t* B_data,
          scalar_t* C_data, int C_size,
    int NC, int NG,
    const int*  M_v,    const int*  N_v,    const int*  K_v,
    const long* A_off,  const int*  A_lda,
    const long* B_off,  const int*  B_ldb,
    const long* C_goff, const int*  C_ldc,
    const int*  G_start)
{
    memset(C_data, 0, (size_t)C_size * sizeof(scalar_t));

    if (NG > 1) {
        #pragma omp parallel for schedule(dynamic)
        for (int g = 0; g < NG; g++) {
            for (int ci = G_start[g]; ci < G_start[g+1]; ci++) {
                int M = M_v[ci], N = N_v[ci], K = K_v[ci];
                int alda = A_lda[ci], bldb = B_ldb[ci], cldc = C_ldc[ci];
                const scalar_t* Ab = A_data + A_off[ci];
                const scalar_t* Bb = B_data + B_off[ci];
                      scalar_t* Cb = C_data + C_goff[ci];
                for (int i = 0; i < M; i++) {
                    const scalar_t* restrict ar = Ab + (long)i * alda;
                          scalar_t* restrict cr = Cb + (long)i * cldc;
                    for (int p = 0; p < K; p++) {
                        scalar_t av = ar[p];
                        const scalar_t* restrict br = Bb + (long)p * bldb;
                        #pragma omp simd
                        for (int j = 0; j < N; j++)
                            cr[j] += av * br[j];
                    }
                }
            }
        }
    } else {
        long out_base = C_goff[0];
        int nt = omp_get_max_threads();
        scalar_t* tbufs = (scalar_t*)calloc((size_t)nt * C_size, sizeof(scalar_t));
        #pragma omp parallel for schedule(dynamic)
        for (int ci = 0; ci < NC; ci++) {
            int tid = omp_get_thread_num();
            int M = M_v[ci], N = N_v[ci], K = K_v[ci];
            int alda = A_lda[ci], bldb = B_ldb[ci], cldc = C_ldc[ci];
            const scalar_t* Ab = A_data + A_off[ci];
            const scalar_t* Bb = B_data + B_off[ci];
                  scalar_t* Cb = tbufs + (size_t)tid * C_size + (C_goff[ci] - out_base);
            for (int i = 0; i < M; i++) {
                const scalar_t* restrict ar = Ab + (long)i * alda;
                      scalar_t* restrict cr = Cb + (long)i * cldc;
                for (int p = 0; p < K; p++) {
                    scalar_t av = ar[p];
                    const scalar_t* restrict br = Bb + (long)p * bldb;
                    #pragma omp simd
                    for (int j = 0; j < N; j++)
                        cr[j] += av * br[j];
                }
            }
        }
        scalar_t* C_out = C_data + out_base;
        for (int t = 0; t < nt; t++) {
            scalar_t* tb = tbufs + (size_t)t * C_size;
            for (long i = 0; i < C_size; i++) C_out[i] += tb[i];
        }
        free(tbufs);
    }
}
