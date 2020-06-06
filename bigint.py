import cupy as cp

# Configurable default values
# Don't change after start!
DISPLAY_DIGITS = 5
SEGMENT_LENGTH = 16
SEGMENT_BASE = 10**SEGMENT_LENGTH

# Constants
MAX_TENSOR_VALUE = 2**63-1



def bigint(x):
    s = str(abs(x))
    s = s.zfill(len(s) + (-len(s) % SEGMENT_LENGTH))

    sign = -1 if x < 0 else 1
    
    tensor = cp.array([
        int(s[i:i + SEGMENT_LENGTH]) * sign
            for i in range(0, len(s), SEGMENT_LENGTH)], dtype=cp.int64)
    
    return BigInt(tensor, sign)

class BigInt:
    def __init__(self, tensor, sign):
        self.tensor = tensor
        self.sign = sign
        
    def __repr__(self):
        string = ''.join(map(str, self.tensor))
        
        if len(string) > DISPLAY_DIGITS * 2:
            return f'{string[:DISPLAY_DIGITS]} ... {string[-DISPLAY_DIGITS:]}'
        else:
            return string
            
    def __add__(a, b):
        a, b = a | b
        
        tensor = a.tensor * a.sign + b.tensor * b.sign
        sign = a.sign if a > b else b.sign
        
        bi = BigInt(tensor * sign, sign)
        bi.carry()
        
        return bi
        
    def __gt__(a, b):
        # Requires implementation Tame Carry
        for x, y in zip(a.tensor, b.tensor):
            x_abs, y_abs = abs(x), abs(y)
            
            if x_abs > y_abs:
                return True
            elif x_abs < y_abs:
                return False
        else:
            return False
        
    def __or__(a, b):
        if a.tensor.size == b.tensor.size:
            return a, b
        
        diff = abs(a.tensor.size - b.tensor.size)
        
        if a.tensor.size < b.tensor.size:
            a.tensor = cp.pad(a.tensor, (diff, 0), 'constant')
        elif b.tensor.size < a.tensor.size:
            b.tensor = cp.pad(b.tensor, (diff, 0), 'constant')
            
        return a, b
            
    def carry(self):
        overflow, tensor = cp.zeros((2, len(self.tensor) + 1))
        tensor[1:] = self.tensor[:]
        
        while True:
            o, t = cp.divmod(tensor, SEGMENT_BASE)
            overflow[:-1], tensor = o[1:], t
            
            if not cp.any(overflow):
                self.tensor = cp.array(tensor, dtype=cp.int64)
                break
                
            tensor = tensor + overflow
        