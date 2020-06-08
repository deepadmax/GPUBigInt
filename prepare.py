# PARSE ARGUMENTS

import argparse

parser = argparse.ArgumentParser(
    description="Prepare ints and bigints for test")

parser.add_argument('-r', '--range', dest='range', nargs=3, type=int, required=True)
parser.add_argument('--pairs', dest='pairs', type=int, default=4)
parser.add_argument('--segment', dest='segment', type=int, default=12)

ns = parser.parse_args()


# CALCULATE INTS AND ARRAYS

# Create directories
import os
import sys

path = '.numbers'
path_tensors  = f'{path}/tensors'
path_integers = f'{path}/integers'

if not os.path.exists(path_tensors):
    os.makedirs(path_tensors)
if not os.path.exists(path_integers):
    os.makedirs(path_integers)

# Set upper limit to entries
R = 10**ns.segment

import cupy as cp
import random

start, end, step = (10**x for x in ns.range)

bytes_length = sys.getsizeof(10**end)

for i in range(start, end+1, step):
    print(f'Preparing for length {i}')

    # Generate random array
    arr = cp.random.randint(0, R, size=(ns.pairs * 2, i // ns.segment))

    # Iterate through all arrays and convert into integers
    integers = set()
    
    for j in range(len(arr)):
        x = int(''.join([
                n.zfill(len(n) + (-len(n) % ns.segment))
                    for n in (str(m) for m in arr[j])]))
                    
        if random.choice([False, True]):
            arr[j] *= -1
            x *= -1
            
        integers.add(x)


    # Save tensor to file
    cp.save(f'{path_tensors}/{i}', arr)
    
    # Save integers to file
    with open(f'{path_integers}/{i}.lsi', 'wb') as f:
        for x in integers:
            f.write(x.to_bytes(bytes_length, byteorder='big', signed=True) + b'\n')