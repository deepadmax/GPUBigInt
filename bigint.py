import tensorflow as tf

class BigInt:
    MAX_DISPLAY_VALUE = 10
    MAX_TENSOR_VALUE = 2**63-1

    def __init__(self, value):
        self.value = value

        # Divide th ebig integer value into a tensor of values within the limit.

    def __repr__(self):
        value_str = str(self.value)
        half_display_value = BigInt.MAX_DISPLAY_VALUE // 2

        if len(value_str) > BigInt.MAX_DISPLAY_VALUE:
            return "%s...%s" % (
                value_str[:half_display_value],
                value_str[:-half_display_value]
            )
        else:
            return value_str

    def __int__(self):
        return self.value

    def __add__(a, b):
        if type(b) is not BigInt:
            if type(b) is int:
                b = BigInt(b)
            else:
                raise TypeError("must be BigInt or int, not %s" % type(b).__name__)

        return BigInt(a.value + b.value)