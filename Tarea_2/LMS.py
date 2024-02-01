import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class LMS():
    def __init__(self, 
                 train_data: pd.DataFrame,
                 test_data: pd.DataFrame,
                 learning_rate: float = 1) -> None:
        
        self.train_data = train_data
        self.test_data = test_data
        self.learning_rate = learning_rate
        self.error_per_epoch = []
        self.weights = np.random.random(len(train_data.columns))

        self.__LMS(self.train_data)
        self.__test_model(self.test_data)

    def __LMS(self, data:pd.DataFrame):
        current_epoch, max_epoch = 0, 500
        threshold = 0.001

        while current_epoch < max_epoch:
            current_error = 0
            for _, row in data.iterrows():
                x = row[data.columns[0: -1]]
                x['b'] = 1
                d = int(row[data.columns[-1]])

                y = np.dot(x, self.weights)

                self.weights = self.weights + self.learning_rate*(d - y)*x

                current_error += (d - y)**2

            current_epoch += 1
            self.error_per_epoch.append(current_error/2)
            if current_epoch > 2 and self.error_per_epoch[-2] - self.error_per_epoch[-1] <= threshold:
                break

        print(f'convergence reached in {current_epoch} epochs')


    def __test_model(self, data: pd.DataFrame): 
        bad_classification_counter = 0
        error = 0
        for _, row in data.iterrows():
            x = row[data.columns[0: -1]]
            x['b'] = 1
            d = int(row[data.columns[-1]])

            y = np.dot(x, self.weights)
            
            error += (d - y)**2

        error = (1/2)*error

        print(f'MSE = {error}')