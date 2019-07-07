from itertools import tee
from typing import Tuple

def pairwise(iterable):
    "s -> (s0,s1), (s1,s2), (s2, s3), ..."
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def between(x: int, bounds: Tuple[int, int]) -> bool:
    return bounds[0] <= x <= bounds[1]
