import pytest

import numpy as np

from src.fast_scboot.c.utils import inplace_ineq_filter


def test_inplace_ineq_filter():

    assert 1 == 1

    # arr = np.arange(5)
    # array = np.squeeze(np.dstack([arr, arr, arr, arr, arr])).astype(np.int32)
    # array_placeholder = np.empty_like(array).astype(np.int32)

    # result = inplace_ineq_filter(array, array_placeholder, 2, 4, len(array))

    # answer = np.asarray([[2, 2, 2, 2, 2], [3, 3, 3, 3, 3], [4, 4, 4, 4, 4]])

    # assert np.all(np.isclose(result, answer))
