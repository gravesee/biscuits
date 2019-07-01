import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from sklearn.datasets import load_breast_cancer

class BasePerformance(ABC):

    @abstractmethod
    def __init__(self, y: pd.Series, w: pd.Series = None):
        if w is None:
            w = np.ones(len(y))
        self.data = pd.DataFrame({'y': y, 'w': w})
        
    @abstractmethod
    def summarize(self, x: pd.Categorical):
        pass

    @abstractmethod
    def plot(self, x: pd.Categorical):
        pass

class BinaryPerformance(BasePerformance):

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
        if y is not None:
            if w is None:
                w = np.ones(len(y))
            data = pd.DataFrame({'y': y, 'w': w})
        else:
            data = self.data


        grps = data.groupby(x)
        aggd = list()
        for _, grp in grps:
            aggd.append(self._aggfun(grp))
        
        res = pd.DataFrame(aggd, index=x.categories)
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
    z = pd.cut(data.data[:,0], cuts)

    p.summarize(z)

# from scipy.sparse import csc_matrix

# dims = (len(z), len(z.categories))
# X = csc_matrix((np.ones(len(z)), (np.arange(0, len(z)), z.codes)), shape=dims)

# from sklearn.linear_model import LogisticRegressionCV

# clf = LogisticRegressionCV(cv=10)

# clf.fit(X, y)

