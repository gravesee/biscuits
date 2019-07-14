import numpy as np
from abc import abstractmethod, ABC
from typing import Optional, Tuple, Type, List
import pandas as pd
from perf import Performance, BinaryPerformance
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

    def register_discretizer(self, name: str, discretizer: Type[Discretizer]):
        self._discretizers[name] = discretizer
    
    def get_discretizer(self, name: str):
        discretizer = self._discretizers.get(name)
        if not name:
            raise ValueError(name)
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


# Register all discretizers here
factory = DiscretizerFactory()
factory.register_discretizer('classifier:decisiontree', DecisionTreeClassifierDiscretizer)
factory.register_discretizer('regression:decisiontree', DecisionTreeRegressorDiscretizer)


if __name__ == "__main__":
    
    data = load_breast_cancer()
    
    X = data.data
    Y = data.target
    for i in range(10):
        X = np.concatenate([X, X], axis=0)
        Y = np.concatenate([Y, Y], axis=0)
    
    Y[1:5000] = 2
    perf = BinaryPerformance(pd.Series(Y))
    
    discretizer = factory.get_discretizer('classifier:decisiontree')
    
    cuts = pd.DataFrame(X).apply(discretizer.discretize, perf=perf, min_samples_leaf=5, max_leaf_nodes=5)

# DecisionTreeClassifier()

    cat = pd.cut(X[:,1], cuts[1])
    res = perf.summarize(cat)

    # pd.value_counts(pd.cut(x, cuts))
        