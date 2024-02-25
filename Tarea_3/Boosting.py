from Neural_Network import Neural_Network
import pandas as pd
from random import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class Boosting:
    def __init__(self,
                 data: pd.DataFrame,
                 number_of_neurons_hidden_layers: list[int],
                 learning_rate: float = 0.01,
                 momentum: float = 0.0,
                 max_epoch: int = 100):
        
        
        N = len(data)
        n1, n2, n3 = N//4, N//3, N//3

        print('::Expert 1')
        n1_data = data.sample(n1, ignore_index=True)
        self.e1 = Neural_Network(data=n1_data,
                                 number_of_neurons_hidden_layers=number_of_neurons_hidden_layers,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 max_epoch=max_epoch)

        print('::Expert 2')
        n2_data = self.__get_n2_data(data, n2)
        self.e2 = Neural_Network(data=n2_data,
                                 number_of_neurons_hidden_layers=number_of_neurons_hidden_layers,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 max_epoch=max_epoch)

        print('::Expert 3')        
        n3_data = self.__get_n3_data(data, n3)
        self.e3 = Neural_Network(data=n3_data,
                                 number_of_neurons_hidden_layers=number_of_neurons_hidden_layers,
                                 learning_rate=learning_rate,
                                 momentum=momentum,
                                 max_epoch=max_epoch)

        self.__plot_data(data)
        
    def __get_n2_data(self, 
                      data: pd.DataFrame,
                      n: int) -> pd.DataFrame:
        n2_examples_index = []
        example_counter = 0

        is_head = random() >= 0.5

        while True:

            for i, row in data.iterrows():
                x = row[data.columns[:-1]]

                y = row[data.columns[-1]]
                
                network_output = self.e1.predict(x)

                if is_head and y != network_output:
                    is_head = random() >= 0.5
                    example_counter += 1
                    n2_examples_index += [i]

                elif not is_head and y == network_output:
                    is_head = random() >= 0.5
                    example_counter += 1
                    n2_examples_index += [i]

                if example_counter >= n:
                    return data.iloc[n2_examples_index]

    def __get_n3_data(self,
                      data: pd.DataFrame,
                      n: int) -> pd.DataFrame:
        n3_examples_index = []
        example_counter = 0

        for i, row in data.iterrows():
            x = row[data.columns[:-1]]
                
            network1_output = self.e1.predict(x)
            network2_output = self.e2.predict(x)

            if network1_output != network2_output:
                example_counter += 1
                n3_examples_index += [i]

            if example_counter >= n:
                return data.iloc[n3_examples_index]
            
        return data.iloc[n3_examples_index]
    
    def predict(self, x: pd.DataFrame) -> int:
        output_1 = self.e1.predict(x)
        output_2 = self.e2.predict(x)
        output_3 = self.e3.predict(x)

        if output_1 == output_2:
            return output_2
        
        return output_3
    

    def __plot_data(self, data: pd.DataFrame):
        committee_y = np.array([])
        for _, row in data.iterrows():
            x = row[data.columns[:-1]]

            committee_output = self.predict(x)

            committee_y = np.append(committee_y, committee_output)

        data['committee_y'] = committee_y
        
        fig, ax = plt.subplots(ncols=2)
        sns.scatterplot(data, x='x1', y='x2', hue='y', ax=ax[0])
        sns.scatterplot(data, x='x1', y='x2', hue='committee_y', ax=ax[1])
        
        ax[0].set_title('Grafico de dispersion del conjunto de datos')
        ax[1].set_title('Grafico de dispersion resultante del comite')
        fig.set_figwidth(10)
        plt.show()


