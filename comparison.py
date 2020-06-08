# PARSE ARGUMENTS

import argparse

parser = argparse.ArgumentParser(
    description="Compare the speed of regular ints and bigints")
    
parser.add_argument('-r', '--range', dest='range', nargs=3, type=int, required=True)
parser.add_argument('-i', dest='iterations', type=int, default=10)

ns = parser.parse_args()


# LOAD INTEGERS AND CUPY ARRAYS

path = '.numbers'
path_tensors  = f'{path}/tensors'
path_integers = f'{path}/integers'

import cupy as cp

from bigint import *

start, end, step = (10**x for x in ns.range)

all_tensors = []
all_integers = []

print('Loading data into memory...')

for i in range(start, end+1, step):
    # Load tensors into bigints
    tensors = cp.load(f'{path_tensors}/{i}.npy')
    tensors = tensors[:len(tensors) - len(tensors) % 2]
    
    a = tuple([BigInt(t) for t in tensors[0::2]])
    b = tuple([BigInt(t) for t in tensors[1::2]])
    
    all_tensors.append(tuple(zip(a, b)))
    
    # Load integers
    with open(f'{path_integers}/{i}.lsi', 'rb') as f:
        integers = [int.from_bytes(x, byteorder='big')  for x in f.readlines()]
        integers = integers[:len(integers) - len(integers) % 2]
        
        x = tuple(integers[0::2])
        y = tuple(integers[1::2])
        
        all_integers.append(tuple(zip(x, y)))
                

# RUN TESTS

import time

plot_cpu, plot_gpu = [], []

print('Starting tests!')

for i, (tensors, integers) in enumerate(zip(all_tensors, all_integers)):        
        
    # GPU with CuPy
    t = time.time()
    
    for u, v in tensors:
        x = u + v
        
    dt = time.time() - t
    plot_gpu.append(dt)
    
    
    # CPU with integers
    t = time.time()
    
    for u, v in integers:
        x = u + v
    
    dt = time.time() - t
    plot_cpu.append(dt)
    
    print(f'Collection {i+1} complete!')


import matplotlib.pyplot as plt

plt.title(f'10^{start}, 10^{end}, 10^{step}')
plt.plot(plot_cpu[1:], 'r', plot_gpu[1:], 'b')
plt.show()






