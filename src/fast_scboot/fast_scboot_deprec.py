

class Sampler:
    
    def __init__(self):
        
        pass
    
    def prepare_data(self, data, stratify_columns, cluster_column):
        
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
        
        self.array = data[['__temp_stratify_column__', '__temp_cluster_column__']].values.astype(np.double)
        
        # self.out = np.empty((int(len(self.data)*1.5), self.data.shape[1]))
        
    def sample_data(self):
        
        num_strats = num_step_unique(self.array[:, 0], len(array))
        num_clusts = num_step_unique(self.array[:, 1], len(array))
        n = len(self.array)
        
        A, B = make_index_matrices(self.array, num_strats, num_clusts, n)
        
        self.arr = np.ascontiguousarray(self.data.values)
        
        samples = np.random.random(size=num_clusts)
        
        sampled_indices = get_sampled_indices(
            samples,
            A,
            B,
            num_strats,
            num_clusts,
            n
        )
        
#         return self.data.iloc[sampled_indices]
        
        inplace_fancy_indexer(self.arr, self.out, sampled_indices, len(sampled_indices), self.arr.shape[1])
        
        return self.out[0:len(sampled_indices)]


        
