import numpy as np
import rpy2.robjects as robjects

from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri
import rpy2.robjects as robj
from utils.region import *

grf = importr("grf")
base = importr("base")
numpy2ri.activate()

__all__ = [
    'ForestModel',
    'update_forest_model',
    'draw_forest_model',
    'greedy_forest_model',
    'fit_multi_forest',
    'predict_multi_forest',
    'predict_multi_forest_oob',
    'forest_muhat_cf',
    'forest_muhat_lfo'
]

class ForestModel:
    """
    Divides the covariate space into regions using regression forests,
    then applies a Thompson Sampling MAB with gaussian prior and likelihood within each region.

    Parameters
    ----------
    K, p: int
        number of arms, covariate dimension

    floor: float
        assignment probability floor

    depth: int
        policy tree depth

    prior_mu, prior_sigma2: float
        If prior_sigma2 small, algorithm's belief in zero-effect is stronger, so convergence is slower.
        As prior_sigma2 tends to inf, we get our usual frequentist / uninformative thompson sampling.

    num_trees: int
        Number of trees used by policy_tree::multi_causal_forest.

    mc: int
        Number of monte-carlo draws to compute thompson probabilities.

    forest_kwargs: dict
        Dictionary of arguments to grf::regression_forest
        e.g. {'num_trees': 2000, 'ci_group_size': 1}

    kasy: bool
        Experimental functionality.
        When enabled, uses Kasy and Sautmann (2019)'s 'exploration sampling'
        by applying the transformation: e_{t}(w) ↦ e_{t}(w)*(1 - e_{t}(w)).
        Note: this is applied *before* the applying the floor.
    """

    def __init__(self,
                 K, p,
                 floor: float = 0.,
                 prior_mu: float = 0.,
                 prior_sigma2: float = 1.,
                 mc: int = 2000,
                 forest_kwargs: dict = None,
                 kasy: bool = False):
        self.K = K
        self.p = p
        self.floor = floor
        self.mc = mc
        self.forest_kwargs = forest_kwargs or {}
        self.kasy = kasy
        assert floor <= 1/K

        # Policy that determines regions
        self.forests = None

        # Prior and posterior parameters
        self._prior_mu = np.full(shape=(K, K), fill_value=prior_mu)
        self._prior_sigma2 = np.full(shape=(K, K), fill_value=prior_sigma2)
        self._mu = np.full(shape=(K, K), fill_value=np.nan)
        self._sigma2 = np.full(shape=(K, K), fill_value=np.nan)

        # Current thompson sampling probabilities
        self.tsprobs = np.full(shape=(K, K), fill_value=np.nan)


def update_forest_model(xs_t, ws_t, yobs_t, probs_t, model: ForestModel):
    """
    Returns a new RegionModel with same configurations but
    updated policy, statistics and Thompson Sampling probabilities
    for each region.
    xs, ws, yobs, probs: np.ndarray
        Input arrays.
    [IMPORTANT] The model is retrained rather than 'updated',
    so inputs must range from time zero up to current time.
    Use: update_region(xs[:t], ws[:t], ...).
    """
    model = deepcopy(model)
    balwts_t = 1 / collect(probs_t, ws_t)

    # Fit new nonparametric policy and predict new region
    model.forests = fr = fit_multi_forest(
        xs=xs_t, ws=ws_t, yobs=yobs_t, K=model.K,
        compute_oob_predictions=True,
        sample_weights=balwts_t,
        **model.forest_kwargs)
    muhat = predict_multi_forest_oob(fr)
    region = np.argmax(muhat, axis=1)

    # Update statistics for each region
    for r, w in product(range(model.K), range(model.K)):
        # Observations in region r that were assigned to arm w
        idx_w = ws_t == w
        idx_r = region == r
        idx_rw = idx_w & idx_r
        n_rw = np.sum(idx_rw)
        # If not enough data, then default to uncond. prob. for each arm
        if n_rw >= 2:
            idx = idx_rw
            n = n_rw
        else:
            idx = idx_w
            n = np.sum(idx_w)

        model._mu[r, w], model._sigma2[r, w] = bayesian_update(
            like_mu=np.mean(yobs_t[idx]),
            like_sigma2=np.var(yobs_t[idx]),
            prior_mu=model._prior_mu[r, w],
            prior_sigma2=model._prior_sigma2[r, w],
            n=n)

    # Updates thompson sampling probabilities for each region
    for r in range(model.K):
        mcdraws = [np.random.normal(model._mu[r], np.sqrt(model._sigma2[r])) for _ in range(model.mc)]
        argmax = np.argmax(mcdraws, axis=1)
        freq = np.bincount(argmax, minlength=model.K).astype(float) / model.mc
        if model.kasy:
            freq = ks_transform(freq)
        model.tsprobs[r] = apply_floor(freq , model.floor)
    return model


def draw_forest_model(xt, model):
    """
    Uses the fitted policy to predict a region,
    then draws from appropriate thompson sampling probabilities.
    """
    xt = np.atleast_2d(xt)
    muhat = predict_multi_forest(model.forests, xt)
    r = np.argmax(muhat, axis=1)
    ps = model.tsprobs[r]
    w = np.array([np.random.choice(model.K, p=p) for p in ps], dtype=np.int_)
    return w, ps


def greedy_forest_model(xt, model):
    """ Pulls the arm with the highest probability within the region """
    xt = np.atleast_2d(xt)
    n = len(xt)
    muhat = predict_multi_forest(model.forests, xt)
    r = np.argmax(muhat, axis=1)
    w = np.argmax(model.tsprobs[r], axis=1)
    ps = np.zeros((n, model.K))
    ps[np.arange(n), w] = 1.
    return w, ps


def forest_muhat_cf(xs, ws, yobs, K, **kwargs):
    """
    Matrix of predictions for each arm. Cross-fitting ensured via OOB.
    """
    forests = fit_multi_forest(xs, ws, yobs, K, **kwargs)
    muhat = predict_multi_forest_oob(forests)
    return muhat


def forest_muhat_lfo(xs, ws, yobs, K, chunks: int, sample_weights=None, **kwargs):
    """
    Fits a sequence of grf::regression_forests sequentially, ensuring that
    each prediction only using past information. To be used for constructing
    doubly-robust scores in an adaptive experiment.

    Fitting and prediction are made in "chunks", so that predictions for
    the bth chunk are computed using the first b-1 chunks. Size of chunks
    is fixed and governed by 'chunks' argument. For the first 'chunk', all
    rows are zero.

    Chunks need not to correspond to batches in an adaptive experiment.
    """
    T = len(xs)
    muhat = np.empty((T, K))
    timepoints = np.arange(start=chunks, stop=T, step=chunks).tolist() + [T]
    ws = ws.reshape(-1, 1)
    yobs = yobs.reshape(-1, 1)
    muhat[:timepoints[0]] = 0
    for t0, t1 in zip(timepoints, timepoints[1:]):
        for w in range(K):
            idx_train = np.where(ws[:t0] == w)[0]
            forest = grf.regression_forest(xs[idx_train], yobs[idx_train], **kwargs)
            pred_future = grf.predict_regression_forest(forest, xs[t0:t1])
            muhat[t0:t1, w] = np.array(pred_future.rx2('predictions'))
    return muhat


def fit_multi_forest(xs, ws, yobs, K, compute_oob_predictions=True, sample_weights=None, **kwargs):
    """
    Fits K grf::regression_forests on data. When compute_oob_predictions is True,
        cross-fitting is ensured via OOB as follows.

    For each arm w:
        * forest = regression_forest(xs[ws == w], yobs[ws == w])
        * muhat[ws == w] = oob_predictions(forest)
        * muhat[ws != w] = predictions(forest, xs[ws != w])

    Note: if you are constructing doubly-robust scores, or won't need
    the forests later, use functions forest_muhat_cf or forest_muhat_lfo instead.
    """
    T = len(xs)
    ws = ws.reshape(-1, 1)
    yobs = yobs.reshape(-1, 1)
    forests = [None] * K
    assert np.issubdtype(ws.dtype, np.integer)

    for w in range(K):
        widx = np.where(ws == w)[0]
        sw = sample_weights[widx] if sample_weights is not None else robj.NULL
        forests[w] = fr = grf.regression_forest(
            X=xs[widx], Y=yobs[widx],
            compute_oob_predictions=compute_oob_predictions,
            sample_weights=sw,
            **kwargs)

        # Keep these indices it the forest is used for cross-fitting.
        if compute_oob_predictions:
            oidx = np.where(ws != w)[0]
            forests[w].widx = widx
            forests[w].oidx = oidx
            forests[w].oob_pred = np.empty(T)
            forests[w].oob_pred[widx] = np.array(grf.predict_regression_forest(fr).rx2('predictions'))
            forests[w].oob_pred[oidx] = np.array(grf.predict_regression_forest(fr, xs[oidx]).rx2('predictions'))
    return forests


def predict_multi_forest_oob(forests):
    """ Retrieves the oob predictions """
    if not hasattr(forests[0], 'oob_pred'):
        raise ValueError("multi_forest was not fit with compute_oob_predictions=True.")
    return np.column_stack([fr.oob_pred for fr in forests])


def predict_multi_forest(forests, xs):
    """
    Predicts a list of forest fit using function fit_multi_forest.
    Note these predictions are NOT oob. Use predict_multi_forest_oob instead.
    """
    T = len(xs)
    K = len(forests)
    muhat = np.empty((T, K))
    for w, fr in enumerate(forests):
        muhat[:, w] = np.array(grf.predict_regression_forest(fr, xs).rx2('predictions'))
    return muhat
