import cupy as cp


# Configurable default values
DISPLAY_DIGITS = 5
SEGMENT_LENGTH = 8

# Constants
MAX_TENSOR_VALUE = 2**63-1


def bigint(x):
    if type(x) is not int:
        raise TypeError('x must be int')
        
    z = str(x)
    z = z.zfill(len(z) + (-len(z) % SEGMENT_LENGTH))
    
    tensor = cp.array([
        int(z[i:i+SEGMENT_LENGTH])
            for i in range(0, len(z), SEGMENT_LENGTH)])
    
    bi = BigInt(tensor)
    
    return bi
    
    
def partition(x):
    z = str(x)
    overflow = int(z[:-SEGMENT_LENGTH].zfill(1))
    value = int(z[-SEGMENT_LENGTH:].zfill(SEGMENT_LENGTH))
    return overflow, value

class BigInt:
    def __init__(self, tensor):
        self.tensor = tensor
        
    def __repr__(self):
        z = ''.join([str(x) for x in self.tensor])
        
        if len(z) > DISPLAY_DIGITS*2:
            return f'{z[:DISPLAY_DIGITS]}..{z[-DISPLAY_DIGITS:]}'
        else:
            return z
            
    def __add__(a, b):
        if type(b) is not BigInt:
            if type(b) is int:
                b = BigInt(b)
            else:
                raise TypeError(f'must be BigInt or int, not {type(b).__name__}')
                
        
        a_t = a.tensor
        b_t = b.tensor
        diff = abs(a_t.size - b_t.size)
        ext = cp.zeros(diff, dtype=cp.int64)
        
        if a_t.size < b_t.size:
            a_t = cp.concatenate((ext, a.tensor))
        elif b_t.size < a_t.size:
            b_t = cp.concatenate((ext, b.tensor))
        
        
a = bigint(1234645159145845616545515615631531)
b = bigint(154151561)
a + b