import numpy as np
from scipy.stats import norm, multivariate_normal
from utils.compute import expand
import warnings
import pandas as pd


def simple_tree_data(T, K=4, p=3, noise_std=1.0, signal=1.0, split=1.676, seed=None, eval_size=10000):
    """
    Splits the covariate space into four regions.
    In each region one of the arms is best on average (see diagram below).

    The larger the 'split' is, the larger the region where arm w=0 is best.
        ie. for more personalization decrease split toward zero.

    Arms w>3 are never best, and covariates x>2 are always noise.

    Default values give optimal/(best_fixed) ratio at 10%.

            ^ x1
            |
        Arm 1 best  |    Arm 3 best
            |       |
    ~~~~~~~~|~(split,split)~~~~~~
            |       |
        Arm 0 best  |    Arm 2 best
    ------(0,0)------------------>x0
            |       |
            |       |
            |       |
            |       |
    """
    assert p >= 2
    assert K >= 4
    assert split >= 0

    rng = np.random.RandomState(seed)
    # Generate experimental data
    xs = rng.normal(size=(T, p))

    r0 = (xs[:, 0] < split) & (xs[:, 1] < split)
    r1 = (xs[:, 0] < split) & (xs[:, 1] > split)
    r2 = (xs[:, 0] > split) & (xs[:, 1] < split)
    r3 = (xs[:, 0] > split) & (xs[:, 1] > split)

    muxs = np.empty((T, K))
    muxs[r0] = np.eye(K)[0]
    muxs[r1] = np.eye(K)[1]
    muxs[r2] = np.eye(K)[2]
    muxs[r3] = np.eye(K)[3]
    muxs = muxs * signal
    ys = muxs + np.random.normal(scale=noise_std, size=(T, K))

    mvn = multivariate_normal([0, 0], np.eye(2))
    mus = np.zeros((K))
    mus[0] = mvn.cdf([split, split])
    mus[1] = mvn.cdf([split, np.inf]) - mvn.cdf([split, split])
    mus[2] = mvn.cdf([split, np.inf]) - mvn.cdf([split, split])
    mus[3] = mvn.cdf([-split, -split])
    mus = mus * signal

    data = dict(xs=xs, ys=ys, muxs=muxs)

    return data, mus


def one_dim_data(T, K=2, p=3, B=2.0, noise_std=1.0, signal=1.0, seed=None):
    """
    """
    assert p >= 1
    assert K == 2

    rng = np.random.RandomState(seed)
    # Generate experimental data
    xs = rng.uniform(low=-B, high=B, size=(T, p))

    r0 = xs[:, 0] < -1
    r1 = (xs[:, 0] >= -1) & (xs[:, 0] < 1)
    r2 = xs[:, 0] >= 1

    muxs = np.empty((T, K))
    muxs[:, 0] = xs[:, 0] ** 2 - 1
    muxs[:, 1] = 1 - xs[:, 0] ** 2
    muxs = muxs * signal
    ys = muxs + np.random.normal(scale=noise_std, size=(T, K))

    mus = np.zeros((K))
    mus[0] = B ** 2 / 3 - 1
    mus[1] = 1 - B ** 2 / 3
    mus = mus * signal

    data = dict(xs=xs, ys=ys, muxs=muxs)

    return data, mus


def generate_bandit_data(X=None, y=None, noise_std=1.0, signal_strength=1.0):
    shuffler = np.random.permutation(len(X))
    x_r = X[shuffler]
    y_r = y[shuffler]
    total_T = min(len(x_r), 30000)
    T = (total_T // 4) * 3
    xs, ys = x_r[:T], y_r[:T]
    xs_test, ys_test = x_r[T:], y_r[T:]
    T, p = xs.shape
    K = len(np.unique(ys))
    muxs = np.array(pd.get_dummies(ys), dtype=float) * signal_strength
    muxs_test = np.array(pd.get_dummies(ys_test),
                         dtype=float) * signal_strength
    ys = muxs + np.random.normal(scale=noise_std, size=(T, K))
    mus = np.bincount(np.array(y, dtype=int)) / T
    data = dict(xs=xs, ys=ys, muxs=muxs, T=T, p=p, K=K, xs_test=xs_test,
                muxs_test=muxs_test, T_test=len(x_r) - T)
    return data, mus
