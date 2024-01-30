import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

class Perceptron():
    def __init__(self, data: pd.DataFrame,
                 algorithm: str,
                 max_iter: int = 100,
                 learning_rate: float = 1) -> None:
        
        self.data = data
        self.algorithm = algorithm
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.error_per_epoch = []
        self.weights = np.random.random(len(data.columns))

        self.__train_model()
    
    def __train_model(self):
        train, test = train_test_split(self.data, train_size=0.8, test_size=0.2)

        if self.algorithm == 'Perceptron':
            self.__perceptron(train)

        elif self.algorithm == 'Averaged':
            self.__averaged(train)
        
        elif self.algorithm == 'MIRA':
            self.__MIRA(train)
        
        else:
            raise Exception('Unexpected algorithm')
        
        self.__test_model(test)


    def __perceptron(self, data: pd.DataFrame):
        current_epoch = 0
        bad_classification_counter = 0
        while True:
            for _, row in data.iterrows():
                x = row[data.columns[0: -1]]
                x['b'] = 1
                d = int(row[data.columns[-1]])

                raw_output = np.dot(x, self.weights)
                y = 1 if raw_output >= 0 else -1
                
                self.weights = self.weights + self.learning_rate*(d - y)*x
                
                if d != y:
                    bad_classification_counter += 1

            self.error_per_epoch.append(bad_classification_counter)
            if bad_classification_counter == 0:
                break

            bad_classification_counter = 0
            current_epoch += 1

        print(f'convergence reached in {current_epoch} epochs')

    def __averaged(self, data: pd.DataFrame):
        current_epoch = 0
        bad_classification_counter = 0
        current_weights = np.array([0] * len(data.columns))
        self.weights = np.array([0] * len(data.columns))
        
        while True:
            for _, row in data.iterrows():
                x = row[data.columns[0: -1]]
                x['b'] = 1
                d = int(row[data.columns[-1]])

                raw_output = np.dot(x, self.weights)
                y = 1 if raw_output >= 0 else -1
                
                
                if d != y:
                    current_weights = current_weights + self.learning_rate*d*x
                    bad_classification_counter += 1

                self.weights += current_weights

            self.error_per_epoch.append(bad_classification_counter)
            if bad_classification_counter == 0:
                break

            bad_classification_counter = 0
            current_epoch += 1

        self.weights = self.weights / current_epoch

        print(f'convergence reached in {current_epoch} epochs')
    
    def __MIRA(self, data:pd.DataFrame):
        current_epoch = 0
        bad_classification_counter = 0
        while True:
            for _, row in data.iterrows():
                x = row[data.columns[0: -1]]
                x['b'] = 1
                d = int(row[data.columns[-1]])

                raw_output = np.dot(x, self.weights)
                y = 1 if raw_output >= 0 else -1
                
                if d != y:
                    self.weights = self.weights + ((d - y) / np.dot(x, x)) * x  
                    bad_classification_counter += 1

            self.error_per_epoch.append(bad_classification_counter)
            if bad_classification_counter == 0:
                break

            bad_classification_counter = 0
            current_epoch += 1

        print(f'convergence reached in {current_epoch} epochs')
    
    def __test_model(self, data: pd.DataFrame): 
        bad_classification_counter = 0

        for _, row in data.iterrows():
            x = row[data.columns[0: -1]]
            x['b'] = 1
            d = int(row[data.columns[-1]])

            raw_output = np.dot(x, self.weights)
            y = 1 if raw_output >= 0 else -1
            
            if d != y:
                bad_classification_counter += 1

        print(f'Number of bad classified examples {bad_classification_counter}')