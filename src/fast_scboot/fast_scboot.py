import time
from collections import defaultdict

import numpy as np

from .c.sample_index_helper import (count_clusts, get_sampled_indices,
                                    make_index_matrix)
from .c.tuple_hash_function import hash_tuple, hash_tuple_2d
from .c.utils import (inplace_fancy_indexer, inplace_ineq_filter,
                      num_step_unique)
from .utils import (get_unique_combinations, read_memmap,
                    record_memmap_metadata, rng_generator, write_memmap)


class Sampler:
    def __init__(self, pre_post=False, use_numpy=True):

        self.pre_post = pre_post
        self.use_numpy = use_numpy

    def prepare_data(self, data, stratify_columns, cluster_column, num_clusts=None):
        """Prepares the data for sampling procedure. The preparation steps are as follows:

        1. Create temporary columns for strata and clusters.
        2. Sort the dataframe by [strata, clusters].
        3. Compute ``_num_clusts``, which is the total number of unique clusters across all strata.
        4. Compute ``_num_strata``, which is the total number of unique strata.
        5. Compute ``_idx_mtx``, which is a matrix with shape [n, 3], whose columns are:

            0: cluster values
            1: start index w.r.t the original data
            2: number of rows of that constitute that cluster

        6. Compute ``_strat_arr``, which is an array of same length as ``_idx_mtx``, that records
            the stratification index.
        7. Compute ``_clust_arr``, which is an array of same length as ``_idx_mtx``, that records
            the cluster index (as opposed to the cluster values).
        8. Compute ``_data_arr``, which is the C-contiguous numpy array of the original dataframe.
        9. Memory map the ``_idx_mtx``, ``_strat_arr``, ``_clust_arr``, ``_data_arr`` and
            delete them from the local memory.

        Parameters
        ----------
        data : Pandas dataframe
            The dataframe with data to sample from. Only numeric data is allowed.

        stratify_columns : str or list of str
            The columns to stratify by.

        cluster_column : str
            The clusters to sample from (from within each stratum).

        num_clusts : int
            The total number of clusters to sample from. If this value is specified, the range of
            cluster values to sample from is first sampled. E.g. if the cluster values are
            [0, 1, 2, 3, 4, 5], and if the num_clusts is set to 3, then the range of clusters to
            sample from is first sampled: e.g. [2, 3, 4]. And the bootstrap sampling of cluster is
            done on this range, e.g. [2, 2, 4].
        """
        self.num_testable_clusts = num_clusts

        if not isinstance(stratify_columns, list):

            stratify_columns = [stratify_columns]

        self.n = len(data)
        self.data_cols = data.columns

        if len(stratify_columns) == 2:

            data["__temp_stratify_column__"] = hash_tuple_2d(
                data[stratify_columns[0]].values.astype(np.double),
                data[stratify_columns[1]].values.astype(np.double),
                self.n,
            )
            data["__temp_stratify_column__"] = (
                data["__temp_stratify_column__"].astype("category").cat.codes
            )

        else:

            data["__temp_stratify_column__"] = hash_tuple(
                np.ascontiguousarray(data[stratify_columns].values.astype(np.double))
            )

        data["__temp_cluster_column__"] = hash_tuple_2d(
            data["__temp_stratify_column__"].values.astype(np.double),
            data[cluster_column].values.astype(np.double),
            self.n,
        )
        data["__temp_cluster_column__"] = (
            data["__temp_cluster_column__"].astype("category").cat.codes
        )

        data = data.sort_values(
            by=["__temp_stratify_column__", "__temp_cluster_column__"]
        )
        data.reset_index(drop=True, inplace=True)
        self.data = data.copy(deep=False)

        arr = self.data["__temp_cluster_column__"].values.astype(np.int32)
        self._num_clusts = num_step_unique(arr, len(arr))

        arr = self.data["__temp_stratify_column__"].values.astype(np.int32)
        self._num_strats = num_step_unique(arr, len(arr))

        tmp_array = data[
            ["__temp_stratify_column__", "__temp_cluster_column__", cluster_column]
        ].values.astype(np.int32)

        tmp_array = np.ascontiguousarray(tmp_array)

        self._idx_mtx, self._strat_arr, self._clust_arr = make_index_matrix(
            tmp_array, self._num_clusts
        )
        # 0: clust_values
        # 1: start_idx
        # 2: nrows

        assert len(self._idx_mtx) == len(self._strat_arr) == len(self._clust_arr)
        self.len_idxs = len(self._idx_mtx)

        # Make it an explicit requirement that the cluster values draw from the same range
        # for each stratum.
        testable_clust_values = self.data[cluster_column].unique()

        # The cluster value that we can start to sample from. If ``num_clusts`` is set
        # to None, then we assume that we will sample from all of available cluster values.
        if self.num_testable_clusts is None:

            self.num_testable_clusts = len(testable_clust_values)

        self.test_startable_clust_values = testable_clust_values[
            : -self.num_testable_clusts + 1
        ]

        self._data_arr = np.ascontiguousarray(data.values)

        if self.use_numpy:

            self.np_metadata = defaultdict(dict)

            record_memmap_metadata(self.np_metadata["_idx_mtx"], self._idx_mtx)
            write_memmap(self.np_metadata["_idx_mtx"], self._idx_mtx)

            record_memmap_metadata(self.np_metadata["_strat_arr"], self._strat_arr)
            write_memmap(self.np_metadata["_strat_arr"], self._strat_arr)

            record_memmap_metadata(self.np_metadata["_clust_arr"], self._clust_arr)
            write_memmap(self.np_metadata["_clust_arr"], self._clust_arr)

            record_memmap_metadata(self.np_metadata["_data_arr"], self._data_arr)
            write_memmap(self.np_metadata["_data_arr"], self._data_arr)

        del self._idx_mtx
        del self._strat_arr
        del self._clust_arr
        del self._data_arr

    def setup_cache(self):
        """Set up the local data to save time on (1) data read and (2) data copy. This 
        is especially useful for parallelization where the main source of the bottleneck 
        is data transfer.
        """
        if not self.pre_post:

            self.idx_mtx_placeholder = np.empty([self.len_idxs, 3]).astype(np.int32)
            self.strat_arr_placeholder = np.empty(self.len_idxs).astype(np.int32)
            self.clust_arr_placeholder = np.empty(self.len_idxs).astype(np.int32)

        else:

            len_idxs = int(self.len_idxs / 2)

            self.idx_mtx_placeholder = np.empty([len_idxs, 3]).astype(np.int32)
            self.strat_arr_placeholder = np.empty(len_idxs).astype(np.int32)
            self.clust_arr_placeholder = np.empty(len_idxs).astype(np.int32)

            self.post_idx_mtx_placeholder = np.empty([len_idxs, 3]).astype(np.int32)
            self.post_strat_arr_placeholder = np.empty(len_idxs).astype(np.int32)
            self.post_clust_arr_placeholder = np.empty(len_idxs).astype(np.int32)

        self._idx_mtx = np.asarray(read_memmap(self.np_metadata["_idx_mtx"]))
        self._strat_arr = np.asarray(read_memmap(self.np_metadata["_strat_arr"]))
        self._clust_arr = np.asarray(read_memmap(self.np_metadata["_clust_arr"]))
        self._data_arr = np.asarray(read_memmap(self.np_metadata["_data_arr"]))

    def sample_data(self, seed=None, out=None):
        """Produce stratified cluster bootstrap sampled data from the original data.
        The sampling algorithm is as follows:

        1. If ``num_clusts`` is used, first get the sampled range of data using the 
        cluster values. The filtering is done inplace to avoid incurring the cost of 
        copying data.
        2. Get the number of unique strata levels and clusters.
        3. Draw sample from [0, 1] uniform distribution ``num_clusts`` number of sample
        points.
        4. For each stratum, multiply the unfirom random value by the the number of 
        clusters within that stratum to map the random variables to the appropriate range
        of natural numbers, which is used to fancy index from the original data. Refer
        to the ``get_sampled_indices`` method.
        """
        start_time = time.time()

        rng = rng_generator(seed)

        test_start_clust_value = rng.choice(self.test_startable_clust_values)
        test_end_clust_value = test_start_clust_value + self.num_testable_clusts - 1

        idx_mtx, strat_arr, clust_arr = inplace_ineq_filter(
            self._idx_mtx,
            self.idx_mtx_placeholder,
            self._strat_arr,
            self.strat_arr_placeholder,
            self._clust_arr,
            self.clust_arr_placeholder,
            test_start_clust_value,
            test_end_clust_value,
            self.len_idxs,
        )

        self.idx_mtx = idx_mtx
        self.strat_arr = strat_arr
        self.clust_arr = clust_arr

        self.num_strats = num_step_unique(strat_arr, len(strat_arr))
        self.num_clusts = num_step_unique(clust_arr, len(clust_arr))

        unif_samples = rng.random(size=self.num_clusts)

        clust_cnt_arr = count_clusts(
            strat_arr, clust_arr, self.num_strats, len(idx_mtx)
        )

        self.clust_cnt_arr = clust_cnt_arr

        sampled_idxs, updated_clust_idxs = get_sampled_indices(
            unif_samples,
            clust_cnt_arr,
            idx_mtx,
            self.num_strats,
            self.num_clusts,
            self.n,
        )

        if out is not None:

            inplace_fancy_indexer(
                self._data_arr,
                out,
                sampled_idxs,
                len(sampled_idxs),
                self._data_arr.shape[1] + 1,  # for the updated_clust_idxs column
                updated_clust_idxs,
            )

            return out[0 : len(sampled_idxs)]

        else:

            pass
