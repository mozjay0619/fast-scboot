from cython cimport boundscheck, cdivision, nogil, nonecheck, wraparound

import numpy as np

cimport numpy as np

np.import_array()

DTYPE = np.float64


@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef double cantor2(double a, double b):
    
    return 0.5 * (a + b) * (a + b + 1) + b

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef double cantor3(double a, double b, double c):
    
    cdef double d = cantor2(a, b)
    
    return 0.5 * (d + c) * (d + c + 1) + c

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef double cantor4(double a, double b, double c, double d):
    
    cdef double e = cantor3(a, b, c)
    
    return 0.5 * (e + d) * (e + d + 1) + d

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
cdef double cantor5(double a, double b, double c, double d, double e):
    
    cdef double f = cantor4(a, b, c, d)
    
    return 0.5 * (f + e) * (f + e + 1) + e

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
def hash_tuple(np.ndarray[DOUBLE_t, ndim=2] _a, int n, int p):
    
    cdef int i
    
    cdef double* a = <double*>(np.PyArray_DATA(_a))
    
    cdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] _b = np.empty(n, dtype=DTYPE, order="C")
    cdef double* b = <double*>(np.PyArray_DATA(_b))
    
    if p==2:

        for i in range(n):

            b[i] = cantor2(a[2*i], a[2*i+1])
            
    elif p==3:
        
        for i in range(n):
            
            b[i] = cantor3(a[3*i], a[3*i+1], a[3*i+2])
            
    elif p==4:
        
        for i in range(n):
            
            b[i] = cantor4(a[4*i], a[4*i+1], a[4*i+2], a[4*i+3])
            
    elif p==5:
        
        for i in range(n):
            
            b[i] = cantor5(a[5*i], a[5*i+1], a[5*i+2], a[5*i+3], a[5*i+4])
        
    return _b


@boundscheck(False)  
@wraparound(False)
@cdivision(True)
def hash_tuple_2d(np.ndarray[DOUBLE_t, ndim=1] _a, np.ndarray[DOUBLE_t, ndim=1] _b, int n):

    cdef int i
    
    cdef double* a = <double*>(np.PyArray_DATA(_a))
    cdef double* b = <double*>(np.PyArray_DATA(_b))

    cdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] _c = np.empty(n, dtype=DTYPE, order="C")
    cdef double* c = <double*>(np.PyArray_DATA(_c))

    for i in range(n):

        c[i] = 0.5 * (a[i] + b[i]) * (a[i] + b[i] + 1) + b[i]

    return _c
