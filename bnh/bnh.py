import numpy as np
import typing
from __future__ import annotations
from typing import Union, Optional, List, Tuple
from util import pairwise
from tabulate import tabulate
from metrics import Metric

class Bin:

    def __init__(self, uniq: np.ndarray, cnts: np.ndarray):
        self.uniq = uniq
        self.cnts = cnts
        pass
    
    def best_split(self, metric: Metric) -> Tuple[float, float]:

        vals = metric.evaluate(self)
        i = metric.best_index(vals)
        
        return self.uniq[i], vals[i]


class BNH:    
    def __init__(
        self,
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
        
        self.bins: List[Bin] = []


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
        self.bins.append(Bin(uniq, cnts))

        self._break_and_heal()

        return (list(uniq), self.exceptions)
        



    
if __name__ == "__main__":
    clf = BNH(1,2, [3])
    