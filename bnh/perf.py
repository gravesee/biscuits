import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, List
from functools import reduce

class Performance(ABC):

    @abstractmethod
    def __init__(self, y: pd.Series, w: pd.Series = None):
        if w is None:
            w = np.ones(len(y))
        self.y = y
        self.w = w
        
    @property
    def values(self) -> Tuple[pd.Series, pd.Series]:
        return (self.y, self.w)
    
    @property
    def data(self) -> pd.DataFrame:
        return pd.DataFrame({'y': self.y, 'w': self.w})
    
    def get_data(self, y: pd.Series = None, w: pd.Series = None) -> pd.DataFrame:
        # pass in a new y
        if y is not None:
            if w is None:
                w = np.ones(len(y))
            else:
                assert(len(y) == len(w))
        # use the existing y, w values
        else:
            y, w = self.values

        return pd.DataFrame({'y': y, 'w': w})

    @abstractmethod
    def _aggregate(self, df: pd.DataFrame) -> pd.DataFrame:
        """ Return summary for perf on group from df.groupby..."""
        pass

    def summarize(self, x: pd.Categorical, y = None, w = None) -> pd.DataFrame:
        data = self.get_data(y, w)
        assert(data.shape[0] == len(x))

        grps = data.groupby(x)
        aggd = list()
        for _, grp in grps:
            aggd.append(self._aggregate(grp))
        
        ## append NaN group
        aggd.append(self._aggregate(data[x.isna()]))
        res = pd.DataFrame(aggd, index=list(x.cat.categories) + ['Missing'])

        return res
        
    @abstractmethod
    def plot(self, x: pd.Categorical):
        pass


class MulticlassPerformance(Performance):

    cols: List[str] = ['N', 'PctN']
    funs: dict = {'Sum': 'sum', 'Pct': lambda y, N: len(y)/N}

    def __init__(self, y: pd.Series, w: pd.Series = None):
        assert(np.issubdtype(y.dtype, np.int64))
        super().__init__(y, w)
        
    def _aggregate(self, df: pd.DataFrame):
        dicts: List[dict] = []
        N = df.shape[0]
        for name, f in self.funs.items():
            res: dict = df.groupby('y')['w'].agg(f, N=N).T.to_dict()
            res = {f"{name}:\n {k}": v for k, v in res.items()}
            dicts.append(res)
        
        res = reduce(lambda x, y: dict(x, **y), dicts, {'N': N})
        return res
    
    def _fix_cols(self, keep):
        for k in self.cols:
            keep.remove(k)
        return self.cols + keep
    
    def summarize(self, x: pd.Categorical, y=None, w=None):
        res: pd.DataFrame = super().summarize(x, y, w)

        # create the toal row
        tot = pd.DataFrame(self._aggregate(self.get_data(y, w)), index=['Total'])
        out: pd.DataFrame = pd.concat([res, tot], sort=False)
        out['PctN'] = out['N'] / len(x)
        
        return out.reindex(columns=self._fix_cols(list(out.keys())))
    
    def plot(self):
        pass
    


class ContinuousPerformance(Performance):

    cols: List[str] = ['N', 'Sum', 'Mean', 'Var']
    
    def __init__(self, y: pd.Series, w: pd.Series = None):
        super().__init__(y, w)
    
    def _aggregate(self, df: pd.DataFrame):
        wy = df['w'] * df['y']
        res = {
            'N': df.shape[0],
            'Sum':  wy.sum(),
            'Mean': wy.mean(),
            'Var':  wy.var()
        }
        return res
    
    def summarize(self, x: pd.Categorical, y=None, w=None):
        res: pd.DataFrame = super().summarize(x, y, w)
        
        # create the toal row
        tot = pd.DataFrame(self._aggregate(self.get_data(y, w)), index=['Total'])
        out: pd.DataFrame = pd.concat([res, tot], sort=False)
        
        return out.reindex(columns=self.cols)
    
    def plot(self):
        pass



class BinaryPerformance(Performance):

    cols: List[str] = ['N', '1s', '0s', 'Rate', 'Pct1', 'Pct0', 'WoE']

    def __init__(self, y: pd.Series, w: pd.Series = None):
        assert(y.isin([0,1]).all())
        super().__init__(y, w)

    def _aggregate(self, df: pd.DataFrame):
        res = {
            'N': df.shape[0],
            '1s': (df['w'][df['y'] == 1]).sum(),
            '0s': (df['w'][df['y'] == 0]).sum()
        }

        return res
    
    def summarize(self, x: pd.Categorical, y=None, w=None):

        res: pd.DataFrame = super().summarize(x, y, w)
        
        # create the toal row
        res = pd.concat([res, res.agg(['sum'])])
        res.rename(index={'sum':'Total'}, inplace=True)

        res['Rate'] = res['1s'] / res['N']
        res['Pct1'] = res['1s'] / res.loc['Total','1s']
        res['Pct0'] = res['0s'] / res.loc['Total','0s']
        res['WoE'] = np.log(res['Pct1'] / res['Pct0'])

        return res.reindex(columns=self.cols)
    
    def plot(self, x):
        pass


if __name__ == "__main__":

    from sklearn.datasets import load_iris


    d = load_iris()
    data = pd.DataFrame(d.data)
    p = MulticlassPerformance(pd.Series(d.target))

    cuts = [-np.inf, 5, 6, 7, np.inf]
    z = pd.Series(pd.cut(d.data[:,0], cuts))

    p.summarize(z)


# from scipy.sparse import csc_matrix

# dims = (len(z), len(z.categories))
# X = csc_matrix((np.ones(len(z)), (np.arange(0, len(z)), z.codes)), shape=dims)

# from sklearn.linear_model import LogisticRegressionCV

# clf = LogisticRegressionCV(cv=10)

# clf.fit(X, y)

