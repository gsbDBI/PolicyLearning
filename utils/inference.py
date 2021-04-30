import numpy as np

from utils.compute import expand
from scipy.stats import norm
import warnings

__all__ = [
    "aw_scores",
    "aw_estimate",
    "aw_var",
    "aw_stderr",
    "aw_tstat",
    "aw_stats",
]


def aw_scores(yobs, ws, balwts, K, muhat=None):
    scores = expand(balwts * yobs, ws, K)  # Y[t]*W[t]/e[t] term
    if muhat is not None:  # (1 - W[t]/e[t])*mu[t,w] term
        scores += (1 - expand(balwts, ws, K)) * muhat
    return scores


def aw_estimate(scores, policy, evalwts=None):
    if evalwts is None:
        evalwts = np.ones((scores.shape[0]))
    return np.sum(evalwts * np.sum(scores * policy, -1)) / np.sum(evalwts) 


def aw_var(scores, estimate, policy, evalwts=None):
    """
    Returns a T x K matrix whose (t, w)-th entry is the *running* standard
    error of arm w-th value estimate at time t:

    std(estimate[t, w]) = sqrt{ sum[s=0 to t] h[s]^2 * (scores[s, w] - estimate[t, w])^2 }
                           --------------------------------------------------------------
                                            sum[s=0 to t] h[s]
    """
    if evalwts is None:
        evalwts = np.ones((scores.shape[0]))
    return np.sum(  (np.sum(policy * scores, -1) - estimate) ** 2    * evalwts ** 2) / (np.sum(evalwts)**2) 


def aw_stderr(scores, estimate, policy, evalwts=None):
    """
    Returns a T x K matrix whose (t, w)-th entry is the *running* standard
    error of arm w-th value estimate at time t:

    std(estimate[t, w]) = sqrt{ sum[s=0 to t] h[s]^2 * (scores[s, w] - estimate[t, w])^2 }
                           --------------------------------------------------------------
                                            sum[s=0 to t] h[s]
    """
    if evalwts is None:
        evalwts = np.ones((scores.shape[0]))
    return np.sqrt(np.sum(  (np.sum(policy * scores, -1) - estimate) ** 2    * evalwts ** 2)) / np.sum(evalwts) 


def aw_tstat(bias, stderr):
    out = bias / stderr
    out[stderr == 0] = np.nan
    return out



def aw_stats(scores, policy, policy_value, evalwts=None):
    """
    Policy value statistics
    """
    if evalwts is None:
        evalwts = np.ones((scores.shape[0]))
    if evalwts.ndim > 1:
        warnings.warn("\nArgument 'mus' must be 1-dimensional."
                      "Raising warning right now, but will raise error "
                      "in the future.")

    # Constant weights by default
    if evalwts is None:
        evalwts = np.ones_like(scores, dtype=np.float_)

    estimate = aw_estimate(scores, policy, evalwts)
    stderr = aw_stderr(scores, estimate, policy, evalwts)
    bias = estimate - policy_value
    tstat = bias / stderr
    cover = (np.abs(tstat) < 1.96).astype(np.float_)
    abserr = np.abs(bias)
    sqerr = bias ** 2
    return np.array([estimate, stderr, bias, cover, tstat, abserr, sqerr, policy_value])


