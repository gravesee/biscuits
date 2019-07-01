import numpy as np
from abc import abstractmethod, ABC
from typing import Optional, Tuple
from bnh.tabulate import tabulate

class DiscretizerMixin(ABC):
    
    @abstractmethod
    def __init__(
        self, 
        x: np.ndarray,
        y: np.ndarray,
        w: Optional[np.ndarray]):
        self.uniq = np.unique(x)
        super().__init__()
    
    @abstractmethod
    def evaluate(self):
        pass
    
    @abstractmethod
    def best_index(self, values: np.ndarray):
        pass
    
    ## TODO:: Pick up here.. should return split value 
    def best_split(self) -> Tuple[int, float]:
        values = self.evaluate()
        i = self.best_index(values)
        return i, values[i]
    

class IVDiscretizerMixin(DiscretizerMixin):
    
    def __init__(self, x, y, w: Optional[np.ndarray] = None):
        uniq, cnts = tabulate(x, y, w)
        self.uniq = uniq
        self.cnts = cnts

    def evaluate(self) -> np.ndarray:        
        tots = self.cnts.sum(axis=0)
        cnta = self.cnts.cumsum(axis=0)
        cntb = tots - cnta
        
        pcta = cnta/tots
        pctb = cntb/tots

        iva = (pcta[:,1] - pcta[:,0]) * np.log(pcta[:,1] / pcta[:,0])
        ivb = (pctb[:,1] - pctb[:,0]) * np.log(pctb[:,1] / pctb[:,0])

        ivs = iva + ivb
        
        return ivs
    
    def best_index(self, values: np.ndarray) -> int:
        return np.argmax(values)
    
        

