import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Perceptron():
    def __init__(self, data: pd.DataFrame,
                 algorithm: str,
                 max_iter: int = 100,
                 learning_rate: float = 1) -> None:
        
        self.data = data
        self.data['independent_variable'] = 1
        
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.error_per_epoch = []


        self.__train_model()
    
    def __train_model(self):
        if self.algorithm == 'Perceptron':
            self.__perceptron()

    def __perceptron(sefl):
        return None