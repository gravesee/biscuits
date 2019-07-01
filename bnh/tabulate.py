import numpy as np
from typing import Optional, Tuple

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

    return uniq, cnts
