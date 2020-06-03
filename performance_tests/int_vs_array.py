import time

import numpy as np
import cupy as cp


START = 10**5
STOP  = 10**6
STEP  = 10**5

SEGMENT_LENGTH = 8

Rmin = 10**(SEGMENT_LENGTH-1)
Rmax = Rmin * 10


plot_cpu, plot_gpu = [], []

for i in range(START, STOP, STEP):
    print(i)
    
    arr = cp.random.randint(Rmin, Rmax, size=i//SEGMENT_LENGTH)
    n = int(''.join(map(str, arr)))
    
    
    # CPU with NumPy
    t = time.time()
    
    a, b = cp.asnumpy(arr), cp.asnumpy(arr)
    x = a + b
    
    dt = time.time() - t
    plot_cpu.append(dt)
    
    
    # GPU with CuPy
    t = time.time()
    
    x = n + n
    
    dt = time.time() - t
    plot_gpu.append(dt)
        
    
import matplotlib.pyplot as plt

plt.plot(plot_cpu[1:], 'r', plot_gpu[1:], 'b')
plt.show()