import cupy as cp

# Configurable default values
# Don't change after start!
DISPLAY_DIGITS = 5
SEGMENT_LENGTH = 12
SEGMENT_BASE = 10**SEGMENT_LENGTH

# Constants
MAX_TENSOR_VALUE = 2**63-1



def bigint(x):
    # Get absolute value to remove minus sign from string
    s = str(abs(x))
    # Pad to match full representation
    s = s.zfill(len(s) + (-len(s) % SEGMENT_LENGTH))

    sign = -1 if x < 0 else 1
    
    # Segment number into smaller parts
    tensor = cp.array([
        int(s[i:i + SEGMENT_LENGTH]) * sign
            for i in range(0, len(s), SEGMENT_LENGTH)], dtype=cp.int64)
    
    return BigInt(tensor)

class BigInt:
    def __init__(self, tensor):
        self.tensor = tensor
        
    
    @classmethod
    def Zeros(cls, size=1):
        return BigInt(cp.zeros(size, dtype=cp.int64))
    
    
    def __repr__(self):
        return str(int(self))
        
    def __str__(self):
        m, s, e = self.components()
        print(m, s, e)
        
        number = str(m)
        
        if len(number) > DISPLAY_DIGITS * 2:
            number = f'{number[:DISPLAY_DIGITS]}...<{len(number)}>...{number[-DISPLAY_DIGITS:]}'
            
        if s < 0:
            return f'{number} - 10^{e}'
        else:
            return number
            
    def __int__(self):        
        m, s, e = self.components()
        # print(m, s, e)
        
        if s < 0:
            return m - 10**e
        else:
            return m
        
            
    def __add__(a, b):
        a.tensor, b.tensor = padsize(a.tensor, b.tensor) # Match their sizes
        
        # Perform heavy carry only if necessary
        try:
            tensor = addnc(a, b)
            
        except OverflowError:
            print('OverflowWarning: Carries a and b')
            a.carry()
            b.carry()
            
            tensor = addnc(a, b)
        
        return BigInt(tensor)
        
    def __sub__(a, b):
        return a + -b

    def __mul__(a, b):
        if b > 0:
            term = a.copy()
            
        elif b < 0:
            term = -a
            b = abs(b)
            
        else:
            return BigInt.Zeros()
        
        # Multiply by adding
        while b := b - 1:
            a = a + term
        return a
        
    def __neg__(self):
        return BigInt(-self.tensor)


    def carry(self):
        overflow, tensor = cp.zeros((2, len(self.tensor) + 1))
        tensor[1:] = self.tensor[:]
        
        # Carry the quotient until no quotients show up
        while True:
            o, t = cp.divmod(tensor, SEGMENT_BASE)
            overflow[:-1], tensor = o[1:], t
            
            if not cp.any(overflow):
                if tensor[0] == 0:
                    tensor = tensor[1:]
                    
                self.tensor = cp.array(tensor, dtype=cp.int64)
                break
                
            tensor = tensor + overflow
            
    def hardcarry(self):
        # Carry right to left using the CPU
        
        overflow = 0
        for i in range(len(self.tensor)-1, -1, -1):
            overflow, self.tensor[i] = divmod(self.tensor[i] + overflow, 10)
        
        ext = []
        if overflow < 0:
            while overflow != -1:
                overflow, remainder = divmod(overflow, 10)
                ext.append(remainder)
            ext.append(overflow)
        
        else:
            while overflow > 0:
                overflow, remainder = divmod(overflow, 10)
                ext.append(remainder)
            
        ext = cp.array(ext[::-1], dtype=cp.int64)
        self.tensor = cp.concatenate((ext, self.tensor))


    def components(self):
        first = str(self.tensor[0])
        subtrahend = int(first[:2]) if first[0] == '-' else 0
        
        self.hardcarry()
        # self.shorten()
        
        
        s = ''.join([
                x.zfill(len(x) + (-len(x) % SEGMENT_LENGTH))
                    for x in (str(abs(x)) for x in self.tensor[abs(subtrahend):])])
                    
        minuend = int(s)
        exponent = len(s)
        
        if subtrahend < 0:
            exponent += len(first[2:])
        
        return minuend, subtrahend, exponent

            
    def shorten(self):
        self.tensor = shorten(self.tensor)
            
    def copy(self):
        return BigInt(cp.copy(self.tensor))


def addnc(a, b):
    # Add two BigInts without carrying
    return a.tensor + b.tensor
        
def padsize(a, b):
    # Make both tensor the same size
    if a.size == b.size:
        return a, b
    
    diff = abs(a.size - b.size)
    
    if a.size < b.size:
        return cp.pad(a, (diff, 0)), b
    elif b.size < a.size:
        return a, cp.pad(b, (diff, 0))
        
def shorten(tensor):
    first = tensor[0]
    
    i = 0
    if first in (0, SEGMENT_BASE-1):
        # Remove any leading zeros or maximum values from a tensor
        for x in tensor:
            if x != first:
                break
            i += 1
        else:
            return BigInt.Zeros()

    if i == 0:
        return tensor
    else:
        return tensor[i:]