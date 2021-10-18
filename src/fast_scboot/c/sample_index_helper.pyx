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
def make_index_matrix(
    np.ndarray[STEP_t, ndim=2] data, 
    int num_clusts):
    """Create a summary matrix with stratification indices and cluster
    indices. This matrix is later used in ``get_sampled_indices`` method.
    The data stored in the array is as follows:

    0: strat_idx
    1: clust_idx
    2: clust_values
    3: start_idx 
        The index w.r.t the original data matrix where the current cluster
        begins. 
    4: nrows
        The number of rows that the current cluster is made of. 

    The outline of how it is used is as follows:

    For each strat_idx, there are multiple clust_idx's. We bootstrap
    sample clust_idx's for each strat_idx. Once we have the bootstrap
    sampled clust_idx's (i.e. bootstrap sampled the rows of the index 
    matrix) we can create the array of index w.r.t the original data
    matrix, with which we can fancy index the data matrix to get the 
    final sampled data.
    """
    cdef int n = data.shape[0]
    cdef int p = data.shape[1]
    
    cdef np.ndarray[STEP_t, ndim=2, mode="c"] _idx_mtx = np.empty([num_clusts, 3], dtype=np.int32, order="C")
    cdef int* idx_mtx = <int*>(np.PyArray_DATA(_idx_mtx))

    cdef np.ndarray[STEP_t, ndim=1, mode="c"] _strat_array = np.empty(num_clusts, dtype=np.int32, order="C")
    cdef int* strat_array = <int*>(np.PyArray_DATA(_strat_array))

    cdef np.ndarray[STEP_t, ndim=1, mode="c"] _clust_array = np.empty(num_clusts, dtype=np.int32, order="C")
    cdef int* clust_array = <int*>(np.PyArray_DATA(_clust_array))
    
    cdef int i, j
    
    cdef int cur_strat_idx = data[0, 0]
    cdef int cur_clust_idx = data[0, 1]  # combined with strat idx
    cdef int cur_clust_val = data[0, 2]  # raw value that can be repeated
    
    cdef int prev_strat_idx = cur_strat_idx
    cdef int prev_clust_idx = cur_clust_idx
    cdef int prev_clust_val = cur_clust_val
    
    cdef int clust_acc = 0
    cdef int start_idx = 0
    
    cdef int pos = 0
    
    for i in range(n):
        
        cur_strat_idx = data[i, 0]
        cur_clust_idx = data[i, 1] 
        cur_clust_val = data[i, 2]
        
        if cur_clust_idx != prev_clust_idx:

            strat_array[pos] = prev_strat_idx
            clust_array[pos] = prev_clust_idx
            
            idx_mtx[pos*3] = prev_clust_val
            idx_mtx[pos*3 + 1] = start_idx
            idx_mtx[pos*3 + 2] = i - start_idx
            
            pos += 1
            start_idx = i
            
        prev_strat_idx = cur_strat_idx
        prev_clust_idx = cur_clust_idx
        prev_clust_val = cur_clust_val

    strat_array[pos] = prev_strat_idx
    clust_array[pos] = prev_clust_idx

    idx_mtx[pos*3] = prev_clust_val
    idx_mtx[pos*3 + 1] = start_idx
    idx_mtx[pos*3 + 2] = i - start_idx + 1

    return _idx_mtx, _strat_array, _clust_array


@boundscheck(False)  
@wraparound(False)
@cdivision(True)
def count_clusts(
    np.ndarray[STEP_t, ndim=1] _strat_array,
    np.ndarray[STEP_t, ndim=1] _clust_array,
    int num_strats,
    int n
    ):
    """The index matrix has data as follows:

    0: strat_idx
    1: clust_idx
    2: clust_values
    3: start_idx 
    4: nrows

    For each strat_idx, there are multiple clust_idx's. The purpose of
    this method is to count how many clust_idx's there are for each
    strat_idx. We want this information because we want to sample with
    replacement that many clust_idx's for each strat_idx. 
    """
    
    cdef np.ndarray[STEP_t, ndim=1, mode="c"] _out = np.empty(num_strats, dtype=np.int32, order="C")
    cdef int* out = <int*>(np.PyArray_DATA(_out))
    
    cdef int* strat_array = <int*>(np.PyArray_DATA(_strat_array))
    cdef int* clust_array = <int*>(np.PyArray_DATA(_clust_array))
    
    cdef int i, acc = 0, clust_acc = 0
    
    cdef double prev_strat_idx = strat_array[0]
    cdef double prev_clust_idx = clust_array[0]
    
    for i in range(n):
        
        if clust_array[i] != prev_clust_idx:
            
            clust_acc += 1
        
        if strat_array[i] != prev_strat_idx:
            
            out[acc] = clust_acc
            
            # Since we are on different strat_idx, restart the clust_idx
            # counting.
            clust_acc = 0
            acc += 1
            
        prev_strat_idx = strat_array[i]
        prev_clust_idx = clust_array[i]
        
    out[acc] = clust_acc + 1
            
    return _out


@boundscheck(False)  
@wraparound(False)
@cdivision(True)
def get_sampled_indices(
    np.ndarray[DOUBLE_t, ndim=1] _unif_samples,
    np.ndarray[STEP_t, ndim=1] _clust_cnt_arr,
    np.ndarray[STEP_t, ndim=2] _idx_mtx,
    int num_strats,
    int num_clusts,
    int n):
    """
    For each stratification level S, there are C_S number of clusters to 
    sample from, where the C_S differs for each S. We want to sample C_S
    clusters for each S.
    
    Given the array ``_unif_samples`` of size ``num_clusts`` of uniform 
    samples, loop through each level of stratification. 
    
    For each level of stratification, loop ``clust_size`` (=C_S) times , 
    where sum of ``clust_size`` for all stratification level equals 
    ``num_clusts``.
    
    Multiply the uniform random number in position of ``clust_size_acc`` 
    (= accumulative sum of C_S's) + current inner loop index by ``clust_size``
    to map the uniform random number to random integer position in ``_idx_mtx``,
    in the partition of the current stratification level. 
    
    Record the randomly chosen cluster's ``start_idx``, ``nrows`` number of
    times in the ``out`` array. This array corresponds to the index positions
    of original data.
    
    Parameters
    ----------
    _unif_samples : ndarray[num_clusts]
        The array of random samples from uniform distribution. 
        
    _clust_cnt_arr : ndarray[num_strats]
        The umber of clusters for each stratification.
        
    _idx_mtx : ndarray[N, 5]
        The index matrix whose columns are:
        
        0: strat_idx
        1: clust_idx
        2: clust_values
        3: start_idx
        4: nrows
        
    num_strats : int
        The number of stratification levels.
        
    num_clusts : int
        The total number of clusters across all stratification levels.
        
    n : int
        The size of original data. 
    """
    
    n *= 2  
    
    cdef double* unif_samples = <double*>(np.PyArray_DATA(_unif_samples))
    cdef int* clust_cnt_arr = <int*>(np.PyArray_DATA(_clust_cnt_arr))
    cdef int* idx_mtx = <int*>(np.PyArray_DATA(_idx_mtx))
    
    # out to store sampled indices
    cdef np.ndarray[STEP_t, ndim=1, mode="c"] _out = np.empty(n, dtype=np.int32, order="C")
    cdef int* out = <int*>(np.PyArray_DATA(_out))

    cdef np.ndarray[STEP_t, ndim=1, mode="c"] _updated_clust_idx = np.empty(n, dtype=np.int32, order="C")
    cdef int* updated_clust_idx = <int*>(np.PyArray_DATA(_updated_clust_idx))
    
    cdef int i, j, k, s, start_idx, nrows, mtx_idx, out_idx = 0
    cdef int clust_size, clust_size_acc = 0

    cdef int updated_clust_idx_acc = 0
    
    for i in range(num_strats):
        
        clust_size = clust_cnt_arr[i]
        
        for j in range(clust_size):
            
            s = <int>(unif_samples[clust_size_acc + j] * clust_size)
            
            mtx_idx = clust_size_acc + s
            
            start_idx = idx_mtx[mtx_idx*3 + 1]
            nrows = idx_mtx[mtx_idx*3 + 2]
            
            for k in range(nrows):

                updated_clust_idx[out_idx] = updated_clust_idx_acc
                
                out[out_idx] = start_idx + k
                out_idx += 1

            updated_clust_idx_acc += 1

        updated_clust_idx_acc = 0
        clust_size_acc += clust_size
        
    return _out[0:out_idx], _updated_clust_idx[0:out_idx]
