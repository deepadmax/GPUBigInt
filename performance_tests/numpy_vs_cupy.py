import time

N = 10
S = (100, 1000, 1000)


# NUMPY with CPU
import numpy as np

t = time.time()

for i in range(N):
    a = np.random.rand(*S)
    b = np.random.rand(*S)
    x_numpy = a + b

dt_numpy = time.time() - t


# CUPY with GPU
import cupy as cp

t = time.time()

for i in range(N):
    a = cp.random.rand(*S)
    b = cp.random.rand(*S)
    x_cupy = a + b

# cp.cuda.Stream.null.synchronize()

dt_cupy = time.time() - t


# PRINT RESULTS
print(f'NumPy: {dt_numpy / N}')
print(f'CuPy:  {dt_cupy  / N}')
print(f'{round((dt_cupy / dt_numpy)*100, 2)}% execution time compared to CPU!')