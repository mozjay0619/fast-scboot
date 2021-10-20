import pytest

import numpy as np

from src.fast_scboot.c.sample_index_helper import (count_clusts,
                                                   make_index_matrix)


def test_make_index_matrix():

    strats = np.asarray([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    clusts = np.asarray([0, 1, 1, 2, 3, 4, 4, 5, 6, 6, 6])
    clust_val = np.asarray([0, 1, 1, 2, 3, 4, 4, 5, 6, 6, 6])
    array = np.squeeze(np.dstack([strats, clusts, clust_val])).astype(np.int32)

    result = make_index_matrix(array, 7)

    answer = np.asarray(
        [
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 2],
            [1, 2, 2, 3, 1],
            [1, 3, 3, 4, 1],
            [1, 4, 4, 5, 2],
            [2, 5, 5, 7, 1],
            [2, 6, 6, 8, 3],
        ]
    )

    assert np.all(np.isclose(result, answer))

    strats = np.asarray([0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
    clusts = np.asarray([0, 1, 1, 2, 3, 4, 4, 5, 6, 6, 7])
    array = np.squeeze(np.dstack([strats, clusts, clusts])).astype(np.int32)

    result = make_index_matrix(array, 8)

    answer = np.asarray(
        [
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 2],
            [1, 2, 2, 3, 1],
            [1, 3, 3, 4, 1],
            [1, 4, 4, 5, 2],
            [2, 5, 5, 7, 1],
            [2, 6, 6, 8, 2],
            [2, 7, 7, 10, 1],
        ]
    )

    assert np.all(np.isclose(result, answer))


def test_count_clusts():

    strats = np.asarray([0, 0, 1, 1, 1])
    clusts = np.asarray([0, 1, 2, 2, 3])
    array = np.squeeze(np.dstack([strats, clusts, clusts, clusts, clusts])).astype(
        np.int32
    )

    result = count_clusts(array, 2, len(array))

    answer = np.asarray([2, 2])

    assert np.all(np.isclose(result, answer))

    strats = np.asarray([0, 0, 1, 1, 1])
    clusts = np.asarray([0, 1, 2, 3, 4])
    array = np.squeeze(np.dstack([strats, clusts, clusts, clusts, clusts])).astype(
        np.int32
    )

    result = count_clusts(array, 2, len(array))

    answer = np.asarray([2, 3])

    assert np.all(np.isclose(result, answer))

    strats = np.asarray([0, 0, 1, 1, 2])
    clusts = np.asarray([0, 1, 2, 3, 4])
    array = np.squeeze(np.dstack([strats, clusts, clusts, clusts, clusts])).astype(
        np.int32
    )

    result = count_clusts(array, 3, len(array))

    answer = np.asarray([2, 2, 1])

    assert np.all(np.isclose(result, answer))

    strats = np.asarray([0, 0, 1, 1, 1])
    clusts = np.asarray([0, 1, 3, 3, 3])
    array = np.squeeze(np.dstack([strats, clusts, clusts, clusts, clusts])).astype(
        np.int32
    )

    result = count_clusts(array, 2, len(array))

    answer = np.asarray([2, 1])

    assert np.all(np.isclose(result, answer))

    strats = np.arange(5)
    clusts = np.arange(5)
    array = np.squeeze(np.dstack([strats, clusts, clusts, clusts, clusts])).astype(
        np.int32
    )

    result = ccount_clusts(array, 5, len(array))

    answer = np.asarray([1, 1, 1, 1, 1])

    assert np.all(np.isclose(result, answer))
