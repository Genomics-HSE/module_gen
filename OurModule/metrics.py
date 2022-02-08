import typing as tp
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


class baseSravnyator:
    ## TODO ОПИСАТЬ ИНТЕРФЕЙС 
    def __init__(self, function) -> None:
        self.function = function
        
    ## @staticmethod
    def __call__(self, *arg) -> tp.Union[int, float, list]:
        return self.function(*arg)
    
    
    
class MSESravnyator(baseSravnyator):
    def __init__(self) -> None:
        super(MSESravnyator, self).__init__(mean_squared_error)
        
class MAESravnyator(baseSravnyator):
    def __init__(self) -> None:
        super(MAESravnyator, self).__init__(mean_absolute_error)
        
        
def KL(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    return np.sum(np.where(a != 0, a * np.log(a / b), 0))


class KLSravnyator(baseSravnyator):
    def __init__(self) -> None:
        super(KLSravnyator, self).__init__(KL)