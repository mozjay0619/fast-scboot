import numpy as np
cimport numpy as np

ctypedef np.npy_float64 DOUBLE_t

cdef double cantor2(double a, double b)
cdef double cantor3(double a, double b, double c)
cdef double cantor4(double a, double b, double c, double d)
cdef double cantor5(double a, double b, double c, double d, double e)