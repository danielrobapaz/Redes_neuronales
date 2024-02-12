import pandas as pd
import numpy as np
from math import exp
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

class Neuron:
    def __init__(self, 
                 number_of_weights: int):
        self.weigths = np.random.random(number_of_weights + 1)
        self.prev_weigths = np.array([0] * (number_of_weights+1))

    def activate(self, input_data: pd.DataFrame) -> float:
        stimulus = np.dot(input_data, self.weigths)
        return self.__sigmoid(stimulus)
    
    def __sigmoid(self, value: float) -> float:
        return 1/(1+exp(-value))


class Neural_Network:
    def __init__(self,
                 data: pd.DataFrame,
                 number_of_neurons_hidden_layers: int = 0,
                 learning_rate: float = 0.01,
                 momentum: float = 0.0,
                 max_epoch: int = 100):
        
        variables = len(data.columns)-1
        
        self.layers = [[Neuron(variables) for _ in range(number_of_neurons_hidden_layers)]]
        self.layers.append([Neuron(number_of_neurons_hidden_layers)])

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_epoch = max_epoch

        train, test = train_test_split(data, train_size=0.8, test_size=0.2, shuffle=True)
        
        self.__train(train)
        self.__plot_error()
        self.__test(test)
        self.__plot_data(data)

    def __feed_fordward(self, x: pd.DataFrame):
        activation_hidden_layer = np.array([])
        
        for i in range(len(self.layers[0])):
            activation = self.layers[0][i].activate(x)
            activation_hidden_layer = np.append(activation_hidden_layer, activation)
        
        activation_hidden_layer = np.append(activation_hidden_layer, 1)

        activation_output_layer = self.layers[1][0].activate(activation_hidden_layer)
        
        return [activation_hidden_layer, activation_output_layer]
    
    def __calculate_gradient(self, 
                             error: float,
                             output: float,
                             hidden_activation: np.array) -> np.array:
        
        gradient_output = (output)*(1-output)*error
        gradient_hidden_layer = []

        output_layer = self.layers[1][0]
        
        for i in range(len(self.layers[0])):
            gradient = hidden_activation[i]*(1-hidden_activation[i])*output_layer.weigths[i]
            gradient_hidden_layer.append(gradient)

        return gradient_hidden_layer, gradient_output

    def __update_neurons(self,
                         input: [float],
                         gradient_hidden_layer: [float],
                         activation_hidden_layer: [float],
                         gradient_output_layer: float):
        
        # update output layer
        for i in range(len(self.layers[1][0].weigths)):
            delta_w = self.learning_rate*gradient_output_layer*activation_hidden_layer[i]
            self.layers[1][0].weigths[i] += delta_w + self.momentum*self.layers[1][0].prev_weigths[i]
            self.layers[1][0].prev_weigths[i] = delta_w + self.momentum*self.layers[1][0].prev_weigths[i]

        # update hidden layers
        for i in range(len(self.layers[0])):
            for j in range(len(self.layers[0][i].weigths)):
                # j_th weight of the i_th neuron in the hidden layer
                delta_w = self.learning_rate*gradient_hidden_layer[i]*input[j]
                self.layers[0][i].weigths[j] += delta_w + self.momentum*self.layers[0][i].prev_weigths[j]
                self.layers[0][i].prev_weigths[j] = delta_w + self.momentum*self.layers[0][i].prev_weigths[j]
        
    def __train(self, data: pd.DataFrame):
        current_epoch = 0
        bad_clasified_examples_per_epoch = []
        while True:
            current_epoch += 1
            bad_clasified_examples = 0
            for _, row in data.iterrows():
                x = row[data.columns[:-1]]
                x['b'] = 1
                d = int(row[data.columns[-1]])
                
                feed_fordward = self.__feed_fordward(x)
                
                activation_hidden_layer = feed_fordward[0]

                activation_output_layer = feed_fordward[1]

                output = 1 if activation_output_layer >= 0.5 else -1
                error = d-output

                if output != d:
                    bad_clasified_examples += 1
                 
                gradient_hidden_layer, gradient_output = self.__calculate_gradient(error,
                                                                                   activation_output_layer,
                                                                                   activation_hidden_layer)
                self.__update_neurons(x,
                                      gradient_hidden_layer,
                                      activation_hidden_layer,
                                      gradient_output)

            percentage_error = 100*bad_clasified_examples/len(data)

            bad_clasified_examples_per_epoch.append(percentage_error)

            if current_epoch >= self.max_epoch:
                y_count = data['y'].value_counts()
                print('::Training::')
                print(f'Examples distribution in training \n {y_count}')
                print(f'Convergence reached in {current_epoch} epochs \nwith {round(percentage_error)}% of error')
                break


            if current_epoch % 20 == 0:
                print(f'Epoch {current_epoch} - Error {percentage_error} %')

        self.error_per_epoch = bad_clasified_examples_per_epoch

    def __test(self, data: pd.DataFrame):
        bad_clasified_examples = 0

        for _, row in data.iterrows():
            x = row[data.columns[:-1]]
            x['b'] = 1
            d = int(row[data.columns[-1]])

            network_output = 1 if self.__feed_fordward(x)[1] >= 0.5 else -1

            if d != network_output:
                bad_clasified_examples += 1
        bad_clasified_examples_percentage = 100*bad_clasified_examples/len(data)
        y_count = data['y'].value_counts()
        
        print('\n::Testing::')
        print(f'Examples distribution in training \n {y_count}')
        print(f'Number of well clasified examples: {bad_clasified_examples} ({round(bad_clasified_examples_percentage, 2)} %)')

    def __plot_data(self, data: pd.DataFrame):
        network_y = np.array([])
        for _, row in data.iterrows():
            x = row[data.columns[:-1]]
            x['b'] = 1

            network_output = 1 if self.__feed_fordward(x)[1] >= 0.5 else -1

            network_y = np.append(network_y, int(network_output))

        data['network_y'] = network_y

        fig, ax = plt.subplots(ncols=2)

        sns.scatterplot(data, x='x1', y='x2', hue='y', ax=ax[0])
        sns.scatterplot(data, x='x1', y='x2', hue='network_y', ax=ax[1])
        
        ax[0].set_title('Grafico de dispersion del conjunto de datos')
        ax[1].set_title('Grafico de dispersion resultante de la red neuronal')
        fig.set_figwidth(10)
        plt.show()

    def __plot_error(self):
        x = np.linspace(1, len(self.error_per_epoch), len(self.error_per_epoch))

        fig, ax = plt.subplots()
        ax.plot(x, self.error_per_epoch)

        ax.set_title('Error (%) por epoca')
        ax.set_xlabel('Epoca')
        ax.set_ylabel('Porcentaje de mal clasificados (%)')
        plt.show()


    def get_output(self, x: pd.DataFrame) -> int:
        return 1 if self.__feed_fordward(x)[1] >= 0.5 else -1