import numpy as np
from utils.forest import forest_muhat_cf

def muhat_random_forest(x, w, y, K, batches=None):
    if batches is None:
        return forest_muhat_cf(x, w, y, K)
    muhat = []
    for b_s, b_e in zip([0, *batches[:-1]], batches):
        m_b = forest_muhat_cf(x[:b_e], w[:b_e], y[:b_e], K)
        muhat.extend(m_b[b_s:])
    return muhat
