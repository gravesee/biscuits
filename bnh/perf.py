import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
from sklearn.datasets import load_breast_cancer

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
    def summarize(self, x: pd.Categorical):
        pass
        
    @abstractmethod
    def plot(self, x: pd.Categorical):
        pass

class BinaryPerformance(Performance):

    def __init__(self, y: pd.Series, w: pd.Series = None):
        assert(y.isin([0,1]).all())
        super().__init__(y, w)

    def _aggfun(self, df: pd.DataFrame):

        res = {
            'N': df.shape[0],
            '1s': (df['w'][df['y'] == 1]).sum(),
            '0s': (df['w'][df['y'] == 0]).sum()
        }

        return res
    
    def summarize(self, x: pd.Categorical, y=None, w=None):

        # TODO: put generalizable stuff in the base class
        data = self.get_data(y, w)
        assert(data.shape[0] == len(x))

        grps = data.groupby(x)
        aggd = list()
        for _, grp in grps:
            aggd.append(self._aggfun(grp))
        
        ## append NaN group
        aggd.append(self._aggfun(data[x.isna()]))
        
        res = pd.DataFrame(aggd, index=list(x.cat.categories) + ['Missing'])
        res = pd.concat([res, res.agg(['sum'])])
        res.rename(index={'sum':'Total'}, inplace=True)

        res['Rate'] = res['1s'] / res['N']
        res['Pct1'] = res['1s'] / res.loc['Total','1s']
        res['Pct0'] = res['0s'] / res.loc['Total','0s']
        res['WoE'] = np.log(res['Pct1'] / res['Pct0'])

        return res

    
    def plot(self, x):
        pass


if __name__ == "__main__":

    data = load_breast_cancer()

    d = load_breast_cancer()
    p = BinaryPerformance(pd.Series(d.target))

    cuts = [-np.inf, 13.0, 13.7, 15.0, 16.9, np.inf]
    z = pd.Series(pd.cut(data.data[:,0], cuts))

    p.summarize(z)


# from scipy.sparse import csc_matrix

# dims = (len(z), len(z.categories))
# X = csc_matrix((np.ones(len(z)), (np.arange(0, len(z)), z.codes)), shape=dims)

# from sklearn.linear_model import LogisticRegressionCV

# clf = LogisticRegressionCV(cv=10)

# clf.fit(X, y)

