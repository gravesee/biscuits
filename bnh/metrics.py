from __future__ import annotations
from typing import Tuple, Type
from abc import ABC, abstractmethod
import numpy as np
from .bin import Bin, BinaryBin


class Metric(ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    @abstractmethod
    def evaluate(bin: Type[Bin]) -> np.ndarray:
        pass
    
    @staticmethod
    @abstractmethod
    def best_index(values: np.ndarray) -> int:
        pass


class IV(Metric):
    """Vectorized information value"""

    @staticmethod
    def evaluate(bin: BinaryBin) -> np.ndarray:        
        tots = bin.cnts.sum(axis=0)
        cnta = bin.cnts.cumsum(axis=0)
        cntb = tots - cnta
        
        pcta = cnta/tots
        pctb = cntb/tots

        iva = (pcta[:,1] - pcta[:,0]) * np.log(pcta[:,1] / pcta[:,0])
        ivb = (pctb[:,1] - pctb[:,0]) * np.log(pctb[:,1] / pctb[:,0])

        ivs = iva + ivb
        
        return ivs
    
    @staticmethod
    def best_index(values: np.ndarray) -> int:
        return np.argmax(values)
