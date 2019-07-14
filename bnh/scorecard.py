# scorecard class
from perf import *
import pandas as pd
import numpy as np
from typing import List, Type
from discretizers import factory, Discretizer
from variable import *

class Scorecard():

    def __init__(self):
        self.variables = {}
        self.performance = None
    
    def discretize(
        self,
        X: pd.DataFrame,
        perf: Type[Performance],
        disc: str,
        exceptions = None, 
        *args,
        **params):

        discretizer = factory.get_discretizer(disc)

        self.performance = perf

        for name, col in X.iteritems():

            ## check type of col here
            # continuous variable
            var = None
            if np.issubdtype(col.dtype, np.number):
                cuts = disc.discretize(col, perf, *args, **params)
                var  = ContinuousVariable(cuts, exceptions)
            else:
                pass

            if var is not None:
                self.variables[name] = var
        
        def display_variable(self, var: Optional[int, str]):
            var = self.variables[var]
            pass
                


if __name__ == "__main__":

    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = pd.DataFrame(data.data)
    y = pd.Series(data.target)

    clf = Scorecard()

    perf = BinaryPerformance(y)
    disc = factory.get_discretizer("BinaryPerformance")

    clf.discretize(X, perf, disc, min_samples_leaf=25, max_depth=4)


    clf.performance.summarize(clf.variables[0].transform(X[0]))

