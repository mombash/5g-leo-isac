"""The selected C++ interpolation operator, loaded from an explicit local build."""
import ctypes
import os
import numpy as np
from .warp_table import interpolation_table

TABLE=interpolation_table()
_library=None


def library():
    global _library
    if _library is None:
        path=os.environ.get('LEO_OFDM_KERNEL_LIBRARY')
        if not path:raise RuntimeError('Set LEO_OFDM_KERNEL_LIBRARY to the explicitly built kernels.so.')
        _library=ctypes.CDLL(path)
        _library.sample_wave.argtypes=[ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int64,ctypes.c_void_p,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
    return _library


def pointer(array):return array.ctypes.data_as(ctypes.c_void_p)


def interpolate(source,coordinates,table=TABLE):
    coordinates=np.ascontiguousarray(coordinates,dtype=np.float64)
    output=np.empty(coordinates.size,dtype=np.complex64)
    library().sample_wave(pointer(source),source.size,pointer(coordinates),coordinates.size,pointer(table),table.shape[0]-1,table.shape[1],pointer(output))
    return output
