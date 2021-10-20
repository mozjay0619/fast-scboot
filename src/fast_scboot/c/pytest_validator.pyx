from cython cimport boundscheck, cdivision, nogil, nonecheck, wraparound

import numpy as np

cimport numpy as np

np.import_array()

ctypedef np.npy_int32 STEP_t
ctypedef np.npy_float32 DTYPE_t
ctypedef np.npy_float64 DOUBLE_t

DTYPE = np.float64


@boundscheck(False)  
@wraparound(False)
@cdivision(True)
def validate_data(np.ndarray[DOUBLE_t, ndim=2] _a):
    
    cdef int n = _a.shape[0]
    cdef int p = _a.shape[1]
    
    cdef int i, j = 0, k = 0
    
    cdef double* a = <double*>(np.PyArray_DATA(_a))
    
    cdef double strat
    cdef double clust
    cdef double cur_clust
    cdef double clust_acc = 0
    
    cdef double prev_clust = a[p-1]
    cdef double prev_strat = a[0]
    
    cdef np.ndarray[DOUBLE_t, ndim=2, mode="c"] _out = np.empty([n, 2], dtype=DTYPE, order="C")
    cdef double* out = <double*>(np.PyArray_DATA(_out))
    
    cdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] _out2 = np.empty(n, dtype=DTYPE, order="C")
    cdef double* out2 = <double*>(np.PyArray_DATA(_out2))
    
    for i in range(n):
        
        cur_clust = a[i*p + p - 1]
        cur_strat = a[i*p]
        
        if (prev_clust != cur_clust) | (prev_strat != cur_strat):
            
            out[j] = a[(i-1)*p + 1]
            out[j + 1] = clust_acc
            j += 2
            
            clust_acc = 1
            
        else:
            
            clust_acc += 1
            
        if prev_strat != cur_strat:
        
            out2[k] = a[i*p - 1] + 1
            k += 1
            
        prev_clust = cur_clust
        prev_strat = cur_strat
        
    out[j] = a[(i)*p + 1]
    out[j + 1] = clust_acc
    
    out2[k] = a[(i+1)*p - 1] + 1
        
    return _out[0:int(j/2)+1], _out2[:k+1]
    