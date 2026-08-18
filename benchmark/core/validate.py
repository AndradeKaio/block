import scipy.io
import numpy as np
from scipy.sparse import csr_matrix

A = scipy.io.mmread("./A.mtx").todense()
B = scipy.io.mmread("./B.mtx").todense()
C = scipy.io.mmread("C.mtx").todense()

C_correct = np.asarray(A @ B)
C = np.asarray(C)
scipy.io.mmwrite("C_correct.mtx", csr_matrix(C_correct))

# TF32 has 10 mantissa bits (~0.1% per op); accumulated over K terms the
# relative error can reach ~1%.  Use rtol=2e-2 to give a comfortable margin.
RTOL = 2e-2
ATOL = 5e-3

diff = np.abs(C_correct - C)
scale = ATOL + RTOL * np.abs(C_correct)
mask = diff > scale

print(f"shape : {C_correct.shape}")
print(f"nnz(C_ref) : {np.count_nonzero(C_correct)}")
print(f"nnz(C_gpu) : {np.count_nonzero(C)}")
print(f"max |err|  : {diff.max():.6g}")
print(f"max rel err: {(diff / (np.abs(C_correct) + 1e-30)).max():.6g}")
print(f"failures   : {mask.sum()} / {(C_correct != 0).sum()} non-zero cells")
print(f"allclose   : {not mask.any()}")
