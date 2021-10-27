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
def num_step_unique(np.ndarray[STEP_t, ndim=1] array, int n):
    
    cdef int i, counter = 1
    cdef double prev = array[0]
    
    for i in range(n):
        
        if prev != array[i]:
            
            counter += 1
        
        prev = array[i]
            
    return counter

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
def inplace_fancy_indexer(np.ndarray[DOUBLE_t, ndim=2] _a, 
                          np.ndarray[DOUBLE_t, ndim=2] _b,
                          np.ndarray[int, ndim=1] _idx, 
                          int n, 
                          int p,
                          np.ndarray[int, ndim=1] _c):
    
    cdef int i, j
    cdef int k = p - 1
    
    cdef double* a = <double*>(np.PyArray_DATA(_a))
    cdef double* b = <double*>(np.PyArray_DATA(_b))
    cdef int* c = <int*>(np.PyArray_DATA(_c))
    cdef int* idx = <int*>(np.PyArray_DATA(_idx))
    
    for i in range(n):
        
        for j in range(k):  # Fill in the i-th row.
            
            b[p*i + j] = a[p*idx[i] + j]

        b[p*i + j + 1] = c[i]

                       
@boundscheck(False)  
@wraparound(False)
@cdivision(True)
def fancy_indexer(np.ndarray[DOUBLE_t, ndim=2] _a, 
                  np.ndarray[int, ndim=1] _idx, 
                  int n, 
                  int p):
    
    cdef int i, j
    
    cdef double* a = <double*>(np.PyArray_DATA(_a))
    cdef int* idx = <int*>(np.PyArray_DATA(_idx))
    
    cdef np.ndarray[DOUBLE_t, ndim=1, mode="c"] _b = np.empty(n*p, dtype=DTYPE, order="C")
    cdef double* b = <double*>(np.PyArray_DATA(_b))
    
    for i in range(n):
        
        for j in range(p):
            
            b[p*i + j] = a[p*idx[i] + j]
    
    return _b.reshape(n, p)

@boundscheck(False)  
@wraparound(False)
@cdivision(True)
def inplace_ineq_filter(np.ndarray[STEP_t, ndim=2] _idx_mtx, 
                        np.ndarray[STEP_t, ndim=2] _idx_mtx_placeholder, 

                        np.ndarray[STEP_t, ndim=1] _strat_arr, 
                        np.ndarray[STEP_t, ndim=1] _strat_arr_placeholder,

                        np.ndarray[STEP_t, ndim=1] _clust_arr, 
                        np.ndarray[STEP_t, ndim=1] _clust_arr_placeholder,

                        int low_inc,
                        int high_inc,
                        int n):
    
    cdef int* idx_mtx = <int*>(np.PyArray_DATA(_idx_mtx))
    cdef int* idx_mtx_placeholder = <int*>(np.PyArray_DATA(_idx_mtx_placeholder))

    cdef int* strat_arr = <int*>(np.PyArray_DATA(_strat_arr))
    cdef int* strat_arr_placeholder = <int*>(np.PyArray_DATA(_strat_arr_placeholder))

    cdef int* clust_arr = <int*>(np.PyArray_DATA(_clust_arr))
    cdef int* clust_arr_placeholder = <int*>(np.PyArray_DATA(_clust_arr_placeholder))
    
    cdef int i, j, val, acc = 0
    
    for i in range(n):
        
        val = idx_mtx[i*3]
        
        if (low_inc <= val) & (val <= high_inc):

            for j in range(3):
                
                idx_mtx_placeholder[acc*3 + j] = idx_mtx[i*3 + j]

            strat_arr_placeholder[acc] = strat_arr[i]
            clust_arr_placeholder[acc] = clust_arr[i]
        
            acc += 1

    return _idx_mtx_placeholder[0:acc], _strat_arr_placeholder[0:acc], _clust_arr_placeholder[0:acc]
