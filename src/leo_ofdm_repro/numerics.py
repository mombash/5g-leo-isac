"""Committed campaign behavior; source mapping in _config/extraction.json."""
import math,struct
Hold=ValueError

def float32_up(value):
    if not math.isfinite(value) or value < 0:
        raise Hold('Threshold must be finite and nonnegative')
    rounded = struct.unpack('<f', struct.pack('<f', value))[0]
    if rounded < value:
        bits = struct.unpack('<I', struct.pack('<f', rounded))[0] + 1
        rounded = struct.unpack('<f', struct.pack('<I', bits))[0]
    return rounded
