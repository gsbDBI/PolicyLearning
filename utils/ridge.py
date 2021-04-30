import numpy as np
from sklearn.linear_model import RidgeCV, Ridge
from sklearn.tree import DecisionTreeRegressor

__all__ = [
    "ridge_init",
    "ridge_update",
    "ridge_muhat_lfo",
    "ridge_muhat_cf",
    "ridge_muhat_lfo_pai",
    "ridge_muhat_lco_pai",
    "ridge_muhat_cf_lfo_pai",
    "tree_muhat_lfo_pai",
]


def ridge_init(p, K):
    # A = np.stack(tuple([np.eye(p + 1)] * K))
    # Ainv = np.stack(tuple([np.eye(p + 1)] * K))
    A = np.empty((K, p + 1, p + 1))
    Ainv = np.empty((K, p + 1, p + 1))
    for k in range(K):
        A[k] = np.eye(p + 1)
        Ainv[k] = np.eye(p + 1)
    b = np.zeros((K, p + 1))
    theta = np.zeros((K, p + 1))
    return A, Ainv, b, theta


def ridge_update(A, b, xt, ytobs):
    xt1 = np.empty(len(xt) + 1)
    xt1[0] = 1.0
    xt1[1:] = xt
    A += np.outer(xt1, xt1)
    b += ytobs * xt1
    Ainv = np.linalg.inv(A)
    theta = Ainv @ b
    return A, Ainv, b, theta


def ridge_muhat_lfo(xs, ws, yobs, K, alpha=1.):
    T, p = xs.shape
    A, Ainv, b, theta = ridge_init(p, K)
    muhat = np.zeros((T, K))
    for t in range(T):
        for w in range(K):
            xt1 = np.empty(p + 1)
            xt1[0] = 1.0
            xt1[1:] = xs[t]
            muhat[t, w] = theta[w] @ xt1
            if ws[t] == w:
                A[w], Ainv[w], b[w], theta[w] = ridge_update(A[w], b[w], xs[t], yobs[t])
    return muhat


def ridge_muhat_lfo_pai(xs, ws, yobs, K, batch_sizes, alpha=1.):
    T, p = xs.shape
    A, Ainv, b, theta = ridge_init(p, K)
    muhat = np.zeros((T, K))

    batch_cumsum = np.cumsum(batch_sizes)
    batch_cumsum = [0] + list(batch_cumsum)

    for l,r in zip(batch_cumsum[:-1], batch_cumsum[1:]):
        for w in range(K):
            # predict from t to T
            xt1 = np.empty((p + 1, r-l))
            xt1[0] = 1.0
            xt1[1:] = xs[l:r].transpose()
            muhat[l:r, w] = theta[w] @ xt1 # dim (T)
        for t in range(l, r):
            w = ws[t]
            A[w], Ainv[w], b[w], theta[w] = ridge_update(A[w], b[w], xs[t], yobs[t])
    return muhat


def ridge_muhat_lco_pai(xs, ws, yobs, K, batch_sizes, alpha=1.):
    T, p = xs.shape
    muhat = np.zeros((T, K))

    batch_cumsum = np.cumsum(batch_sizes)
    batch_cumsum = [0] + list(batch_cumsum)

    for l,r in zip(batch_cumsum[:-1], batch_cumsum[1:]):
        xs_b = np.concatenate((xs[:l], xs[r:]), axis=0)
        ws_b = np.concatenate((ws[:l], ws[r:]), axis=0)
        yobs_b = np.concatenate((yobs[:l], yobs[r:]), axis=0)
        x1 = np.empty((r-l, p+1))
        x1[:, 0] = 1.0
        x1[:, 1:] = xs[l:r]
        
        for w in range(K):
            T_w = np.sum(ws_b==w)
            if T_w == 0:
                theta = np.zeros(p+1)
            else:
                x = np.concatenate((np.ones((T_w, 1)), xs_b[ws_b==w]), axis=1)
                xTx = np.matmul(x.transpose(), x)
                xTy = x.transpose().dot(yobs_b[ws_b==w])
                theta = np.linalg.solve(xTx, xTy)
            muhat[l:r, w] = x1.dot(theta)
    return muhat


def tree_muhat_lfo_pai(xs, ws, yobs, K, batch_sizes):
    T, p = xs.shape
    muhat = np.zeros((T, T, K))

    batch_cumsum = np.cumsum(batch_sizes)
    for l,r in zip(batch_cumsum[:-1], batch_cumsum[1:]):
        xl, wl, yl = xs[:l], ws[:l], yobs[:l]
        for w in range(K):
            regr = DecisionTreeRegressor(max_depth=3)
            regr.fit(xl[wl==w], yl[wl==w])
            muhat[l:r, :, w] = regr.predict(xs)
    return muhat


def ridge_muhat_cf_lfo_pai(xs, ws, yobs, K, batch_sizes, cv=5, nfolds=5):
    T, p = xs.shape
    muhat = np.zeros((T, T, K))
    batch_cumsum = np.cumsum(batch_sizes)
    batch_cumsum = list(batch_cumsum)

    muhat = np.zeros((T, T, K))
    for l,r in zip(batch_cumsum[:-1], batch_cumsum[1:]):
        ws_l = np.copy(ws)
        ws_l[l:] = -1
        muhat[l:r, :, :] = ridge_muhat_cf(xs, ws_l, yobs, K, cv=cv, nfolds=nfolds)

    return muhat


def ridge_muhat_cf(xs, ws, yobs, K, cv=5, nfolds=5):
    T, p = xs.shape
    muhat = np.empty((T, K))
    ridgecv = RidgeCV(cv=cv)

    for w in range(K):
        # Predict on obs that *were not* assigned w
        widx = np.where(ws == w)[0]
        oidx = np.where(ws != w)[0]
        muhat[oidx, w] = ridgecv.fit(xs[widx], yobs[widx]).predict(xs[oidx])
        ridge = Ridge(alpha=ridgecv.alpha_)

        # Cross-fit on obs that *were* assigned to w
        np.random.shuffle(widx)
        folds = np.array_split(widx, nfolds)
        for f in range(nfolds):
            wfold = folds[f]
            wnonfold = np.hstack([*folds[:f], *folds[f + 1:]])
            muhat[wfold, w] = ridge.fit(xs[wnonfold], yobs[wnonfold]).predict(xs[wfold])

    return muhat
