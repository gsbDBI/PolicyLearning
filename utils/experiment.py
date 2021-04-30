from utils.thompson import *
from utils.region import *
from functools import partial
import numpy as np
from copy import deepcopy


def run_experiment(xs, ys, floor_start, floor_decay, bandit_model, batch_sizes, recorded_T=None):
    T, K = ys.shape
    _, p = xs.shape
    ws = np.empty(T, dtype=np.int_)
    yobs = np.empty(T)
    probs = np.zeros((T, T, K))
    Probs_t = np.zeros((T, K))

    if bandit_model == 'TSModel':
        agent = LinTSModel(K=K, p=p, floor_start=floor_start,
                           floor_decay=floor_decay)
    elif bandit_model == 'RegionModel':
        agent = RegionModel(T=T, K=K, p=p, floor_start=floor_start,
                            floor_decay=floor_decay, prior_sigma2=.1)
    elif bandit_model == 'kasy-RegionModel':
        agent = RegionModel(T=T, K=K, p=p, floor_start=floor_start,
                            floor_decay=floor_decay, prior_sigma2=.1, kasy=True)

    # uniform sampling at the first batch
    batch_size_cumsum = list(np.cumsum(batch_sizes))
    if recorded_T is not None:
        recorded_idx = [np.where(np.array(batch_size_cumsum)>=TT)[0][0] 
                for TT in recorded_T]
    ws[: batch_size_cumsum[0]] = np.arange(batch_size_cumsum[0]) % K
    yobs[: batch_size_cumsum[0]] = ys[np.arange(
        batch_size_cumsum[0]), ws[: batch_size_cumsum[0]]]
    probs[: batch_size_cumsum[0]] = 1/K
    Probs_t[: batch_size_cumsum[0]] = 1/K
    if bandit_model.endswith('RegionModel'):
        agent = update_region(xs[:batch_size_cumsum[0]],
                              ws[:batch_size_cumsum[0]
                                 ], yobs[:batch_size_cumsum[0]],
                              Probs_t[:batch_size_cumsum[0]], 1, agent)
    elif bandit_model == 'TSModel':
        agent = update_thompson(xs[:batch_size_cumsum[0]],
                                ws[:batch_size_cumsum[0]
                                   ], yobs[:batch_size_cumsum[0]],
                                agent)

    recorded_agents = []
    for idx, (f, l) in enumerate(zip(batch_size_cumsum[:-1], batch_size_cumsum[1:]), 1):
        if bandit_model.endswith('RegionModel'):
            w, p = draw_region(
                xs=xs, model=agent, start=f, end=l)
        else:
            w, p = draw_thompson(
                xs=xs, model=agent,
                start=f, end=l, current_t=f)
        yobs[f:l] = ys[np.arange(f, l), w]
        ws[f:l] = w
        probs[f:l, :] = np.stack([p] * (l-f))
        Probs_t[f:l] = p[f:l]
        if bandit_model.endswith('RegionModel'):
            agent = update_region(
                xs[:l], ws[:l], yobs[:l], Probs_t[:l], idx+1, agent)
        elif bandit_model == 'TSModel':
            agent = update_thompson(xs[f:l], ws[f:l], yobs[f:l], agent)
        if recorded_T is not None and idx in recorded_idx:
            recorded_agents.append(deepcopy(agent))

    data = dict(yobs=yobs, ws=ws, xs=xs, ys=ys,
                probs=probs, recorded_agents=recorded_agents)

    return data
