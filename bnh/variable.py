from __future__ import annotations
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Container, Type, Optional
import pandas as pd
from scipy.sparse import csc_matrix
import numpy as np
from itertools import chain
from perf import BinaryPerformance
from util import between


class Variable(ABC):
    
    def transform(self, x: pd.Series, type='categorical'):
        
        if type is 'categorical':
            return self._to_categorical(x)
        elif type is 'sparse':
            return self._to_sparse(x)
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
    def expand(self, i: int, values: List[float]):
        pass
        
class CategoricalVariable(Variable):
    
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
        """combine levels provided by idx into one level"""
        # filter to which levels are selected in idx
        lvls = [l for i, l in enumerate(self.levels) if i in idx]
        
        keys = list()
        for k, v in self.map.items():
            if ','.join(v) in lvls:
                keys.append(k)
        
        for k in keys:
            self.map[k] = keys
    
    def expand(self, i: int, values: List[float] = None):
        """explode combined levels into original members"""
        lvls = self.levels
        if not between(i, (0, len(lvls))):
            return
        
        for k, v in self.map.items():
            if ','.join(v) == lvls[i]:
                self.map[k] = [k]
            

class ContinuousVariable(Variable):

    @staticmethod
    def __pad_inf(x: List[float]) -> List[float]:
        """Make list unique and bookend with inf"""
        x.insert(0, -np.inf)
        x.append(np.inf)
        x = sorted(list(set(x)))
        return x

    def __init__(self, cuts: List[float], exceptions: List[float] = None):
        self.cuts = self.__pad_inf(cuts)
        if exceptions is None:
            exceptions = []
        self.exceptions = exceptions
    
    def _to_categorical(self, x: pd.Series) -> pd.Categorical:
        res = pd.cut(x, self.cuts)
        res = res.cat.add_categories(self.exceptions)
        for e in self.exceptions:
            res[x.isin([e])] = e
        
        return res
    
    def collapse(self, idx: Tuple[int, int]):
        """remove requested cutpoints"""
        cuts = [x for j, x in enumerate(self.cuts) if not between(j, idx)]
        self.cuts = self.__pad_inf(cuts)
    
    def expand(self, i: int, values: Optional[List[float]] = None):
        """Replace requested index in cuts with new cut points"""
        if values is None:
            values = self.cuts[i:(i+1)]

        cuts = self.cuts.copy()
        cuts[i:i] = values
        self.cuts = self.__pad_inf(cuts)


if __name__ == "__main__":

    x = np.floor(np.random.random_sample(1_000_000) * 10)
    x = pd.Series(x)

    # x = pd.Series([-1,-2,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3, np.nan])
    b = ContinuousVariable([3.,7.], exceptions=[0,1])
    b.expand(0, [1.])
    
    res1 = b._to_categorical(x)
    b._to_sparse(x)

    y = x.astype(str)
    bc = CategoricalVariable(y.unique())

    bc.collapse([1,3])
    res2 = bc._to_categorical(y)
    bc._to_sparse(y)

    y = pd.Series(np.floor(np.random.random_sample(1_000_000) * 2))
    perf = BinaryPerformance(y)

    b.collapse((0,2))
    perf.summarize(x=b._to_categorical(x))


