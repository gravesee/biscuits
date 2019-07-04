from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Container, Type
import pandas as pd
from scipy.sparse import csc_matrix
import numpy as np
from itertools import chain

# TODO: move to util
def between(x: int, bounds: Tuple[int, int]) -> bool:
    return bounds[0] <= x <= bounds[1]


class Bin(ABC):
    
    def transform(self, x: pd.Series, type='categorical'):
        
        if type is 'categorical':
            self._to_categorical(x)
        elif type is 'sparse':
            self._to_sparse(x)
        else:
            ValueError("type must be in ['categorical', 'sparse']")

    
    @abstractmethod
    def _to_categorical(self, x: pd.Series) -> pd.Categorical:
        pass
    
    #@abstractmethod
    def collapse(self, idx: Container[int]):
        pass
    
    def _to_sparse(self, x: pd.Series) -> csc_matrix:
        z = self._to_categorical(x)
        # create information for sparse matrix (data, (rows, cols))
        
        data = np.ones(len(x))
        rows = np.arange(0, len(x), dtype=int)
        
        cols = z.cat.codes
        cols[cols == -1] = len(z.cat.categories)
        
        ncat = len(z.cat.categories) + 1
        return csc_matrix((data, (rows, cols)), shape=(len(x), ncat + 1))

    @abstractmethod
    def expand(self, i: int):
        pass

class BinCategorical(Bin):
    
    def __init__(self, categories: List[str]):
        self.map: Dict[str, List[str]] = {x: [x] for x in categories}
    
    def _to_categorical(self, x: pd.Categorical) -> pd.Categorical:
        map = {k: ','.join(v) for k, v in self.map.items()}
        res = x.replace(map)
        res =  pd.Categorical(res, categories=list({x: None for x in map.values()}))
        return pd.Series(res)
    
    @property
    def levels(self):
        # each level maps to a list of levels
        # get all of the unique values of the levels
        return list({','.join(l): None for l in self.map.values()})
    
    def collapse(self, idx: List[int]):
        # filter to which levels are selected in idx
        lvls = [l for i, l in enumerate(self.levels) if i in idx]
        
        keys = list()
        for k, v in self.map.items():
            if ','.join(v) in lvls:
                keys.append(k)
        
        for k in keys:
            self.map[k] = keys
    
    def expand(self, idx: int):
        lvls = self.levels
        if not between(idx, (0, len(lvls))):
            return
        
        for k, v in self.map.items():
            if ','.join(v) == lvls[idx]:
                self.map[k] = [k]
            


class BinContinuous(Bin):

    def __init__(self, cuts: List[float], exceptions: List[float] = []):
        cuts = sorted(list(set(cuts)))
        cuts.insert(0, -np.inf)
        cuts.append(np.inf)
        self.cuts = cuts
        self.exceptions = exceptions
    
    def _to_categorical(self, x: pd.Series) -> pd.Categorical:
        res = pd.cut(x, self.cuts)
        res = res.cat.add_categories(self.exceptions)
        for e in self.exceptions:
            res[x.isin([e])] = e
        
        return res
    
    def collapse(self, idx: Tuple[int, int]):
        cuts = self.cuts
        cuts = [x for j, x in enumerate(cuts) if not between(j, idx)]
    
    def expand(self, idx: int):
        pass


if __name__ == "__main__":
    x = pd.Series([-1,-2,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3, np.nan])
    b = BinContinuous([1.], exceptions=[-1,-2])
    
    res1 = b._to_categorical(x)
    b._to_sparse(x)

    y = pd.Series(['a','b','c','d','a','b','c','d','e'])
    bc = BinCategorical(['a','b','c','d'])

    bc.collapse([1,3])
    res2 = bc._to_categorical(y)
    type(res2)
    bc._to_sparse(y)

