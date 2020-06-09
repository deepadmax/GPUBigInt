# PARSE ARGUMENTS

import argparse

parser = argparse.ArgumentParser(
    description="Compare the speed of regular ints and bigints")
    
parser.add_argument('-r', '--range', dest='range', nargs=3, type=int, required=True)
parser.add_argument('-c', '--carry', dest='carry', type=int, default=-1)
parser.add_argument('-l', '--log',   dest='log',   type=int, default=100000)

ns = parser.parse_args()



# LOAD INTEGERS AND CUPY ARRAYS

import sys

path = '.numbers'
path_tensors  = f'{path}/tensors'
path_integers = f'{path}/integers'

import os
from random import shuffle

import cupy as cp

from bigint import *

start, end, step = ns.range

all_tensors = []
all_integers = []

bytes_length = sys.getsizeof(10**end)

print('Loading data into memory...')

for i in range(start, end+1, step):
    path_tensor  = f'{path_tensors}/{i}.npy'
    path_integer = f'{path_integers}/{i}.lsi'
    
    # Skip if missing numbers
    if not os.path.exists(path_tensor) or not os.path.exists(path_integer):
        print(f'Could not find tensors or integers for {i}')
        continue
        
    # Load tensors into bigints
    tensors = cp.load(path_tensor)
    shuffle(tensors)
    
    if len(tensors) % 2 != 0:
        tensors.pop()

    a = tuple([BigInt(t) for t in tensors[0::2]])
    b = tuple([BigInt(t) for t in tensors[1::2]])
    
    tensors = tuple(zip(a, b))
    
    # Load integers
    with open(path_integer, 'rb') as f:
        integers = [int.from_bytes(x, byteorder='big', signed=True) 
                        for x in f.read().split(b'[END_OF_INTEGER]') if x]
        shuffle(integers)
        
        if len(integers) % 2 != 0:
            integers.pop()

        x = tuple(integers[0::2])
        y = tuple(integers[1::2])
        
        integers = tuple(zip(x, y))
        
    if len(tensors) != len(integers):
        max_len = min(len(tensors), len(integers))
        tensors  = tensors [:max_len]
        integers = integers[:max_len]
    
    all_tensors.append(tensors)
    all_integers.append(integers)



# RUN TESTS

import time

plot_gpu, plot_cpu = [], []
t_gpu, t_cpu = 0, 0

print('Starting tests!')

for i, (tensors, integers) in enumerate(zip(all_tensors, all_integers)):  
    j = start + i*step
    
    # Skip if missing numbers
    if tensors is None or integers is None:
        continue
    
    
    # GPU with CuPy
    t = time.time()
    
    for u, v in tensors:
        x = u + v
    
    if ns.carry > 0 and j % ns.carry == 0:
        x.carry()
        
    dt = time.time() - t
    t_gpu += dt
    plot_gpu.append((j, dt / len(tensors)))
    
    # CPU with integers
    t = time.time()
    
    for u, v in integers:
        x = u + v
    
    dt = time.time() - t
    t_cpu += dt
    plot_cpu.append((j, dt / len(integers)))
    
    if ns.log > 0 and j % ns.log == 0:
        print(f'Batch {j} tested')

print(f'GPU total time: {t_gpu}')
print(f'CPU total time: {t_cpu}')



# PLOT BENCHMARK

print('Now plotting...')

import matplotlib.pyplot as plt

plt.title(f'{start}, {end}, {step}')

plot_gpu_x, plot_gpu_y = zip(*plot_gpu[1:])
plot_cpu_x, plot_cpu_y = zip(*plot_cpu[1:])

plt.plot(plot_gpu_x, plot_gpu_y, 'b', label='GPU', linestyle='-', marker=' ')
plt.plot(plot_cpu_x, plot_cpu_y, 'r', label='CPU', linestyle='-', marker=' ')

plt.xlabel('length in digits')
plt.ylabel('time in seconds')

plt.show()






