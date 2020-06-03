import cupy as cp


# Configurable default values
# Don't change after start!
DISPLAY_DIGITS = 5
SEGMENT_LENGTH = 8
SEGMENT_BASE = 10**SEGMENT_LENGTH

# Constants
MAX_TENSOR_VALUE = 2**63-1


def bigint(x):
    if type(x) is not int:
        raise TypeError('x must be int')
        
    z = str(x)
    
    if x < 0:
        return bigint(0) - bigint(abs(x))
        
    z = z.zfill(len(z) + (-len(z) % SEGMENT_LENGTH))
    
    tensor = cp.array([
        int(z[i:i+SEGMENT_LENGTH])
            for i in range(0, len(z), SEGMENT_LENGTH)])
    
    return BigInt(tensor)
    
def match_size(a, b):
    diff = abs(a.size - b.size)
    ext = cp.zeros(diff, dtype=cp.int64)
    
    if a.size < b.size:
        a = cp.concatenate((ext, a))
    elif a.size > b.size:
        b = cp.concatenate((ext, b))
        
    return a, b

class BigInt:
    def __init__(self, tensor, sign=0):
        if tensor[0] < 0:
            self.sign = -1
            self.tensor = tensor[1:]
                
        else:
            self.sign = 1
            self.tensor = tensor
            
        if sign != 0:
            self.sign = sign
            
        self.carry()
        
    def __repr__(self):
        z = ''.join([str(x) for x in self.tensor])
        
        if len(z) > DISPLAY_DIGITS*2:
            return f'{z[:DISPLAY_DIGITS]}..{z[-DISPLAY_DIGITS:]}'
        else:
            return z
            
    def __int__(self):
        value = int(''.join([str(x) for x in self.tensor]))
        
        if self.sign == -1:
            return 10**len(self.tensor) - value
        else:
            return value
            
    def __add__(a, b):        
        a_t, b_t = match_size(a.tensor, b.tensor)
        tensor = a_t + b_t
        
        if a.sign == -1 and b.sign == -1:
            return BigInt(tensor, sign=-1)
            
        return BigInt(tensor)
        
    def __sub__(a, b):
        a_t, b_t = match_size(a.tensor, b.tensor)
        
        if a.sign == -1 and b.sign == -1:
            return BigInt(a_t + b_t, sign=-1)
        
    def carry(self):        
        overflow = 0
        
        for i in range(len(self.tensor)-1, -1, -1):
            self.tensor[i] += overflow
            
            if self.tensor[i] < 0:
                overflow = -1
                self.tensor[i] = SEGMENT_BASE - self.tensor[i]
                
            elif self.tensor[i] >= SEGMENT_BASE:
                overflow = 1
                self.tensor[i] -= SEGMENT_BASE
                
            else:
                overflow = 0
        
        
        self.sign = overflow
        
        if overflow == 1:
            self.tensor = cp.concatenate((cp.array([1]), self.tensor))
        elif overflow == -1:
            self.sign = -1
            
            
# a = bigint(1234)
# print(int(a))
# 
# b = bigint(23456)
# print(int(b))
# 
# c = a + b
# print(c.tensor)