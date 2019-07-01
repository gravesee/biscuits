from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from bnh.metrics import Metric
from typing import Tuple, Type

class Bin(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def best_split(self, metric: Type[Metric]) -> Tuple[float, float]:
        pass


class BinaryBin(Bin):

    def __init__(self, uniq: np.ndarray, cnts: np.ndarray):
        self.uniq = uniq
        self.cnts = cnts
    
    def best_split(self, metric: Type[Metric]) -> Tuple[float, float]:

        vals = metric.evaluate(self)
        i = metric.best_index(self)
        
        return self.uniq[i], vals[i]

