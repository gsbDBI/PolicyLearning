import numpy as np
from itertools import product
from itertools import chain

__all__ = ['collect',
           'collect3',
           'expand',
           'draw',
           'apply_floor',
           'stick_breaking',
           'ks_transform',
           'preprocess',
           'crossfit_indices']

def collect3(arr):
    T, T2,K = arr.shape
    assert(T==T2)
    out = np.empty((T, K), dtype=arr.dtype)
    for t in range(T):
        out[t] = arr[t, t, :]
    return out


def collect(arr, idx):
    out = np.empty(len(idx), dtype=arr.dtype)
    for i, j in enumerate(idx):
        out[i] = arr[i, j]
    return out


def expand(values, idx, num_cols):
    out = np.zeros((len(idx), num_cols), dtype=values.dtype)
    for i, (j, v) in enumerate(zip(idx, values)):
        out[i, j] = v
    return out


def draw(p):
    return np.searchsorted(np.cumsum(p), np.random.random(), side='right')


def apply_floor(a, amin):
    new = np.maximum(a, amin)
    total_slack = np.sum(new) - 1
    individual_slack = new - amin
    c = total_slack / np.sum(individual_slack)
    return new - c * individual_slack


def stick_breaking(Z):
    T, K = Z.shape
    weights = np.zeros((T, K))
    weight_sum = np.zeros(K)
    for t in range(T):
        weights[t] = Z[t] * (1 - weight_sum)
        weight_sum += weights[t]
    return weights


def preprocess(x, w, K):
    t = len(x)
    w_dummy = expand(np.ones(t), w, K)[:, 1:]
    kx = x.shape[1]
    kw = w_dummy.shape[1]
    xw = np.column_stack([x[:, i] * w_dummy[:, j] for i, j in product(range(kx), range(kw))])
    return np.column_stack([x, w, xw])


def crossfit_indices(T: int, num_folds: int):
    """
    Return tuples of (train, test) indices.
    """
    idx = [x.tolist() for x in np.array_split(range(T), num_folds)]
    for i in range(num_folds):
        train = idx[i]
        test = list(chain(*(idx[:i] + idx[i + 1:])))
        yield train, test


def ks_transform(p: np.ndarray, axis: int = 0) -> np.ndarray:
    """
    Applies Kasy and Sautmann (2019)'s 'exploration sampling'
    transformation: e_{t}(w) ↦ e_{t}(w)*(1 - e_{t}(w))

    >>> ks_transform([0., 1., 0.])
    array([0.33333333, 0.33333333, 0.33333333])

    >>> ks_transform([[.2, .5, .3], [.1, .1, .8]], axis=1)
    array([[0.25806452, 0.40322581, 0.33870968],
           [0.26470588, 0.26470588, 0.47058824]])
    """
    safety_eps = 1e-5
    if not np.all(np.isclose(np.sum(p, axis=axis), 1.)):
        raise ValueError(f'Input does not sum to one on axis {axis}:\n{p}')
    p = np.clip(p, safety_eps, 1 - safety_eps)
    p_ks = p * (1 - p)
    p_ks = p_ks / np.sum(p_ks, axis=axis, keepdims=True)
    return p_ks
