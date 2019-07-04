import numpy as np
from abc import abstractmethod, ABC
from typing import Optional, Tuple, Type, List
import pandas as pd
from bnh.perf import Performance, BinaryPerformance
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_breast_cancer

class Discretizer(ABC):
    @abstractmethod
    def discretize(self, x: pd.Series, perf: Type[Performance], *args, **kwargs) -> List[float]:
        pass

class DiscretizerFactory():
    """Class that takes an input variable and perf object and returns a set of cuts"""

    def __init__(self):
        self._discretizers = {}

    def register_discretizer(self, perf: str, discretizer: Type[Discretizer]):
        self._discretizers[perf] = discretizer
    
    def get_discretizer(self, perf: str):
        discretizer = self._discretizers.get(perf)
        if not perf:
            raise ValueError(perf)
        return discretizer()

class DecisionTreeDiscretizer(ABC):

    @abstractmethod
    def __init__(self, clf):
        self.clf = clf
    
    def discretize(self, x: pd.Series, perf: Type[Performance], *args, **params):
        clf = self.clf
        clf.set_params(**params)

        y, w = perf.values
        clf.fit(x.values.reshape(-1, 1), y, sample_weight=w)

        tree = clf.tree_
        res = [v for v, x in zip(tree.threshold, tree.feature) if x != -2]
        
        ## add -np.Inf and np.inf
        res.insert(0, -np.inf)
        res.append(np.inf)
        res.sort()
        return res

class DecisionTreeClassifierDiscretizer(DecisionTreeDiscretizer, Discretizer):

    def __init__(self):
        self.clf = DecisionTreeClassifier()

class DecisionTreeRegressorDiscretizer(DecisionTreeDiscretizer, Discretizer):

    def __init__(self):
        self.clf = DecisionTreeRegressor()


factory = DiscretizerFactory()
factory.register_discretizer('BinaryPerformance', DecisionTreeClassifierDiscretizer)
factory.register_discretizer('ContinuousPerformance', DecisionTreeRegressorDiscretizer)

def discretize(x, perf, discretizer=None, **params):
    if discretizer is None:
        discretizer = type(perf).__name__
    disc = factory.get_discretizer(discretizer)
    return disc.discretize(x, perf, **params)


if __name__ == "__main__":
    
    data = load_breast_cancer()
    #x = pd.DataFrame(data.data[:,0])
    #y = pd.Series(data.target)
    #perf = BinaryPerformance(y)
    
    # cuts = discretize(x, perf, min_samples_leaf=5, max_leaf_nodes=5)

    X = data.data
    Y = data.target
    for i in range(10):
        X = np.concatenate([X, X], axis=0)
        Y = np.concatenate([Y, Y], axis=0)
    perf = BinaryPerformance(pd.Series(Y))
    
    
    cuts = pd.DataFrame(X).apply(discretize, perf=perf, min_samples_leaf=5, max_leaf_nodes=5)




    # DecisionTreeClassifier()

    cat = pd.cut(X[:,1], cuts[1])
    res = perf.summarize(cat)

    # pd.value_counts(pd.cut(x, cuts))
        