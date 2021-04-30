import numpy as np

__all__ = ['bayesian_update']


def bayesian_update(
    like_mu: np.ndarray,
    like_sigma2: np.ndarray,
    prior_mu: np.ndarray,
    prior_sigma2: np.ndarray,
    n: int):
    """
    Posterior with normal prior, normal likelihood.
    Ref: http://www.ams.sunysb.edu/~zhu/ams570/Bayesian_Normal.pdf (page 4)
    """
    post_sigma2 = 1 / (1 / prior_sigma2 + n / like_sigma2)
    post_mu = post_sigma2 * (prior_mu / prior_sigma2 + n * like_mu / like_sigma2)
    return post_mu, post_sigma2
