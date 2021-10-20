import itertools

import numpy as np
import pytest

from src.fast_scboot.fast_scboot import Sampler
from src.fast_scboot.utils import rng_generator


def flatten_lists(list_of_lists):

    return list(itertools.chain.from_iterable(list_of_lists))


def fake_data_generator(seed=None, max_strats=100, max_clusts=100):

    rng = rng_generator(seed)

    num_strats = rng.choice(np.arange(3, max_strats))
    num_clusts = rng.choice(np.arange(3, max_clusts), size=num_strats)

    num_clusts = [max(i, 2) for i in num_clusts]
    max_clust = max(num_clusts) - 1

    clusts = [
        [np.repeat(i, i) for i in range(1, num_clust)] for num_clust in num_clusts
    ]
    clust_array = np.hstack(flatten_lists(clusts))

    _strats = [sum([len(elem) for elem in _clusts]) for _clusts in clusts]
    strats = np.arange(len(_strats))
    strat_array = np.repeat(strats, _strats)

    data = np.squeeze(np.dstack([strat_array, clust_array])).astype(np.double)
    data = pd.DataFrame(data, columns=["strat", "clust"])

    return data, max_clust, num_clusts


def test_sampler():

    for i in range(100):

        data, max_clust, num_clusts = fake_data_generator(seed=i)

        s = Sampler()
        s.prepare_data(data, "strat", "clust", num_clusts=max_clust)
        s.setup_cache()
        out = np.empty([data.shape[0] * 2, data.shape[1] + 1])
        res = s.sample_data(seed=i, out=out)

        out1, out2 = validate_data(res)

        assert all(np.equal(num_clusts, out2 + 1))
        assert all(np.equal(out1[:, 0], out1[:, 1]))
