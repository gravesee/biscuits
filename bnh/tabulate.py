import numpy as np
from typing import Optional, Tuple
from bnh import Bin

def tabulate(
    x: np.ndarray,
    y: np.ndarray,
    w: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    
    if w is None:
        w = np.ones(len(x))

    uniq = np.unique(x)
    ux = uniq[..., np.newaxis] # needed for broadcasting
    cnts = np.array(tuple(((x == ux)[:,y == v]).dot(w[y == v]) for v in [0,1])).T

    return (uniq, cnts)

def iv(lower: np.ndarray, upper: np.ndarray, tots: np.ndarray):
    """Vectorized information value"""
    # assert(len(cnts) == 2)
    # assert(len(tots) == 2)

    a = lower/tots
    b = upper/tots

    iva = (a[:,1] - a[:,0]) * np.log(a[:,1] / a[:,0])
    ivb = (b[:,1] - b[:,0]) * np.log(b[:,1] / b[:,0])

    return iva + ivb

def best_split(bin: Bin) -> Tuple[float, float]:
    tots = bin.table.sum(axis=0)
    lower = bin.table.cumsum(axis=0)
    upper = tots - lower

    ivs = iv(lower, upper, tots)

    i = np.argmax(ivs)
    
    return bin.uniq[i], ivs[i]
