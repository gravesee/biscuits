from __future__ import annotations
import numpy as np
from typing import Union, Optional, List, Tuple, Type, ClassVar
from bnh.util import pairwise
from bnh.tabulate import tabulate
from bnh.bin import Bin
from discretizers import DiscretizerMixin



class BNH(DiscretizerMixin):
    def __init__(
        self,
        bintype: Bin, 
        max_bins: int = 5,
        min_obs_per_bin: int = 5,
        exceptions: List[float] = None,
        epsilon_break: float = .001,
        epsilon_heal: float = .01):
        
        """DOC STRING"""

        if exceptions is None:
            exceptions = []
        
        self.max_bins = max_bins
        self.min_obs_per_bin = max_bins
        self.exceptions = exceptions
        self.epsilon_break = epsilon_break
        self.epsilon_heal = epsilon_heal
        self.bintype = bintype
        self.bins: List[Type[Bin]] = []


    def _break_bin(self, bin) -> List[Bin]:
        pass

    def _merge_bins(self, binx: Bin, biny: Bin) -> Bin:        
        pass

    ## should break 
    def _break(self):
        for bin in self.bins:
            self._break_bin(bin)

    ## then heal using min values
    def _heal(self):
        # iter pairs here ... 
        for (binx, biny) in pairwise(self.bins):
            self._merge_bins(binx, biny)

    def _break_and_heal(self):
        # bounce back and forth here
        pass
    
    def _fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        w: np.ndarray = None
    ) -> Tuple[List[float], List[float]]:
        """Return tuple of optimal cuts and exceptions"""

        # filter exceptions and nans
        # f = np.isin

        uniq, cnts = tabulate(x, y, w)

        ## add tje omotoa; nom
        self.bins.append(bintype.)

        self.bins.append(Bin(uniq, cnts))

        self._break_and_heal()

        return (list(uniq), self.exceptions)
        




### TODO: Follow this approach

from abc import ABC, abstractmethod

class DiscretizerMixin(ABC):
    def __init__(self, x, y, w):
        uniq, cnts = -1, [x,y,w]
        self.uniq = uniq
        self.cnts = cnts
        super().__init__()
    
    def find_best_split(self):
        return 100

class BreakAndHeal(DiscretizerMixin):
    def __init__(self, opts, x=1, y=2, w=3):
        self.opts = opts
        super().__init__(x, y, w)


from sklearn.tree import DecisionTreeClassifier, 
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

def disco(x, y, **kwargs):
    clf = DecisionTreeClassifier(**kwargs)
    clf.fit(x.reshape([-1,1]), y)
    return clf

x = data.data
y = data.target

for i in range(10):
    x = np.concatenate([x,x], axis=0)
    y = np.concatenate([y,y], axis=0)

clfs = [disco(x, y, max_leaf_nodes=5) for x in np.rollaxis(x, 1)]

cuts = [sorted([t for t in clf.tree_.threshold if t != -2]).insert(0, np.NINF) for clf in clfs]

cuts = [sorted([t for t in clf.tree_.threshold if t != -2]) for clf in clfs]
for cut in cuts:
    cut.insert(0, -np.Inf)
    cut.append(np.Inf)


np.apply_along_axis(disco, 1, data.data, y=data.target)

import pandas as pd

pd.DataFrame([pd.cut(x, c) for x, c in zip(np.rollaxis(x, 1), cuts)])
