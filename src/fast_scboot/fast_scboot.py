from .c.tuple_hash_function import hash_tuple
from .c.sample_index_helper import count_clusts, get_sampled_indices, make_index_matrix
from .c.utils import num_step_unique, inplace_fancy_indexer, inplace_ineq_filter
from .utils import get_unique_combinations, rng_generator
import numpy as np
import time


class Sampler:
    
    def __init__(self):
        
        pass
    
    def prepare_data(self, data, stratify_columns, cluster_column, num_clusts):
        
        self.num_clusts = num_clusts
        
        if not isinstance(stratify_columns, list):
            stratify_columns = [stratify_columns]

        stratify_columns_numeric = []
        tmp_columns = []

        for stratify_column in stratify_columns:

            if data[stratify_column].dtype == "O":

                temp_codified_column = stratify_column + "__codified__"
                data[temp_codified_column], _ = get_encoding(data[stratify_column])

                stratify_columns_numeric.append(temp_codified_column)
                tmp_columns.append(temp_codified_column)

            else:

                stratify_columns_numeric.append(stratify_column)
                
        data['__temp_stratify_column__'] = get_unique_combinations(
            data[stratify_columns_numeric].values)
        
        data['__temp_cluster_column__'] = get_unique_combinations(
            data[['__temp_stratify_column__', cluster_column]].values)
        
        data = data.sort_values(by=[
            '__temp_stratify_column__', '__temp_cluster_column__'])
        
        data.reset_index(drop=True, inplace=True)
        self.data = data.copy(deep=False)
        
        arr = self.data['__temp_cluster_column__'].values.astype(np.int32)
        self._num_clusts = num_step_unique(arr, len(arr))
        
        arr = self.data['__temp_stratify_column__'].values.astype(np.int32)
        self._num_strats = num_step_unique(arr, len(arr))
        
        self.array = data[['__temp_stratify_column__', 
                           '__temp_cluster_column__', 
                           cluster_column]].values.astype(np.int32)
        
        self.array = np.ascontiguousarray(self.array)

        self.n = len(self.data)
        
        self.idx_mtx, self.strat_arr, self.clust_arr = make_index_matrix(self.array, self._num_clusts)

        # 0: clust_values
        # 1: start_idx
        # 2: nrows
        
        assert len(self.idx_mtx) == len(self.strat_arr) == len(self.clust_arr)
        self.len_idxs = len(self.idx_mtx)
        
        # set up cache
        self.idx_mtx_placeholder = np.empty([self.len_idxs, 3]).astype(np.int32)
        self.strat_arr_placeholder = np.empty(self.len_idxs).astype(np.int32)
        self.clust_arr_placeholder = np.empty(self.len_idxs).astype(np.int32)
        
        # Expi particular codes
        testable_clust_values = self.data[cluster_column].unique()
        self.test_startable_clust_values = testable_clust_values[:-self.num_clusts+1]
        
        self.data_arr = np.ascontiguousarray(data.values)
        
    def sample_data(self, seed=None, out=None):
        
        rng = rng_generator(seed)
        
        test_start_clust_value = rng.choice(self.test_startable_clust_values)
        test_end_clust_value = test_start_clust_value + self.num_clusts - 1
        
        idx_mtx, strat_arr, clust_arr = inplace_ineq_filter(
            self.idx_mtx,
            self.idx_mtx_placeholder, 
            self.strat_arr,
            self.strat_arr_placeholder,
            self.clust_arr,
            self.clust_arr_placeholder,
            test_start_clust_value, 
            test_end_clust_value, 
            self.len_idxs
        )

        self.num_strats = num_step_unique(strat_arr, len(strat_arr))
        self.num_clusts = num_step_unique(clust_arr, len(clust_arr))

        unif_samples = np.random.random(size=self.num_clusts)
        
        clust_cnt_arr = count_clusts(
            strat_arr,
            clust_arr,
            self.num_strats,
            len(idx_mtx)
        )
        
        sampled_idxs, updated_clust_idxs = get_sampled_indices(
            unif_samples,
            clust_cnt_arr,
            idx_mtx,
            self.num_strats,
            self.num_clusts,
            self.n
        )
        
        if out is not None:
            
            inplace_fancy_indexer(
                self.data_arr, 
                out, 
                sampled_idxs, 
                len(sampled_idxs), 
                self.data_arr.shape[1] + 1,  # for the updated_clust_idxs column
                updated_clust_idxs
            )

            return out[0:len(sampled_idxs)]
        
        else:
            
            pass
        
        
