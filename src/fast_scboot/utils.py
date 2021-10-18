import tempfile

import numpy as np


def rng_generator(seed=None, n=1):
    """Numpy random Generator generator.

    Parameters
    ----------
    seed : int or None
        Random seed.

    n : int
        The number of Generators to return.

    Returns
    -------
    if n==1:
        Generator
    if n>1:
        list of Generators

    Reference
    ---------
    https://numpy.org/doc/stable/reference/random/generator.html#numpy.random.Generator
    https://towardsdatascience.com/stop-using-numpy-random-seed-581a9972805f
    https://albertcthomas.github.io/good-practices-random-number-generators/
    """
    if n > 1:

        ss = np.random.SeedSequence(seed)
        child_seeds = ss.spawn(n)
        rng_streams = [np.random.default_rng(s) for s in child_seeds]
        return rng_streams

    else:

        return np.random.default_rng(seed)


def get_unique_combinations(a):
    """Create an array of indices that are unique to the combination of the values from
    each column of the given array. Refer to the examples below.

    Parameters
    ----------
    a : numpy array

    Returns
    -------
    b : numpy array

    Examples
    --------
    a = np.asarray([0, 0, 1, 1])
    b = get_unique_combinations(a)
    b

    >>> array([0, 0, 1, 1])

    a = np.asarray([[0, 1],
                    [0, 0],
                    [0, 1]])
    b = get_unique_combinations(a)
    b

    >>> array([1, 0, 1])

    a = np.asarray([[0, 1, 0],
                    [1, 0, 0],
                    [0, 1, 2],
                    [1, 0, 0],
                    [2, 0, 0],
                    [0, 1, 3]])
    b = get_unique_combinations(a)
    b

    >>> array([ 10, 100,  12, 100, 200,  13])
    """
    if isinstance(a, list):
        a = np.asarray(a)

    if len(a.shape) == 1:
        a = a.reshape(-1, 1)

    multipliers = [1]
    for i in range(a.shape[1] - 1, 0, -1):
        n = len(np.unique(a[:, i]))
        digits = int(np.log10(n)) + 1
        multipliers.append(10 ** (digits))

    # https://stackoverflow.com/questions/20787484/fast-replacement-of-numpy-values
    _, inv = np.unique(a, return_inverse=True)
    new_indices = inv.reshape(a.shape)

    multiplier = np.asarray(multipliers)
    multiplier = np.cumprod(multiplier)
    multiplier = multiplier[::-1]

    return np.sum(new_indices * multiplier, axis=1)


def write_memmap(metadata, array):

    writable_memmap = np.memmap(
        metadata["filepath"],
        dtype=metadata["dtype"],
        shape=metadata["shape"],
        mode="w+",
    )

    writable_memmap[:] = array[:]
    del writable_memmap


def read_memmap(metadata, idx=None):

    readonly_memmap = np.memmap(
        metadata["filepath"],
        dtype=metadata["dtype"],
        shape=metadata["shape"],
        mode="r",
    )

    if idx is None:
        array = readonly_memmap[:]
    else:
        if is_nested_list(idx):
            array = readonly_memmap[tuple(idx)]
        else:
            array = readonly_memmap[idx]

    del readonly_memmap
    return array


def record_memmap_metadata(d, array):

    d["shape"] = array.shape
    d["dtype"] = str(array.dtype)
    d["filepath"] = tempfile.TemporaryFile()
