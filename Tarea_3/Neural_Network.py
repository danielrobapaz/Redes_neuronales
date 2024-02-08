import pandas as pd
import numpy as np
from math import exp, log2, floor
from itertools import combinations_with_replacement

class Neuron:
    def __init__(self, 
                 number_of_weights: int):
        self.weigths = np.random.random(number_of_weights + 1)

    def activate(self, input_data: pd.DataFrame) -> float:
        stimulus = np.dot(input_data, self.weigths)
        return self.__sigmoid(stimulus), stimulus
    
    def __sigmoid(self, value: float) -> float:
        return 1/(1+exp(-value))
    
    def activate_derivate(self, value: float) -> float:
        return self.__sigmoid(value)*(1-self.__sigmoid(value))


class Neural_Network:
    def __init__(self,
                 data: pd.DataFrame,
                 layers: [int] = []):
        
        variables = len(data.columns)-1
        self.layers = []
        for i, number_of_neurons in enumerate(layers):
            if i == 0:
                self.layers.append([Neuron(variables) for _ in range(number_of_neurons)])

            else:
                self.layers.append([Neuron(layers[i-1]) for _ in range(number_of_neurons)])

        self.answer_map = [[1, 0], [0, 1]]

        self.__train(data)
        self.__test()

    def __feed_fordward(self, x: pd.DataFrame):
        activation = []
        stimulus = []
        for i in range(len(self.layers)):
            current_activation = []
            current_stimulus = []

            for j in range(len(self.layers[i])):
                #j_th neuron of the i_th layer
                if i == 0:
                    # input layer
                    act, stim = self.layers[i][j].activate(x) 
                else:
                    # hidden layers
                    act, stim = self.layers[i][j].activate(activation[i-1])
                
                current_activation .append(act)
                current_stimulus.append(stim)

            if i != len(self.layers)-1:
                current_activation += [1]

            activation.append(np.array(current_activation))
            stimulus.append(np.array(current_stimulus))

        return activation, stimulus
    
    def __calculate_gradient(self, 
                             error: np.array,
                             stimulus: np.array,
                             activation: np.array) -> np.array:
        gradient = []
        for i in range(len(self.layers)-1, -1, -1):
            current_gradient = []
            for j in range(len(self.layers[i])):
                #j_th neuron of the i_th layer
                derivate = self.layers[i][j].activate_derivate(stimulus[i][j])
                if i == len(self.layers)-1:
                    # output layer
                    neuron_gradient = -error[j]*derivate*activation[i][j]

                else:
                    # hidden layer
                    next_layer_weights = np.array([n.weigths[j] for n in self.layers[i+1]])
                    neuron_gradient = derivate*np.dot(next_layer_weights, gradient[0])
                    
                current_gradient.append(neuron_gradient)
            
            gradient.insert(0, np.array(current_gradient))

        return gradient

    def __train(self, 
                data: pd.DataFrame):
        for _, row in data.iterrows():
            x = row[data.columns[:-1]]
            x['b'] = 1
            d = int(row[data.columns[-1]])
            
            activation, stimulus = self.__feed_fordward(x)

            output = activation[-1]
            expected_output = self.answer_map[d]
            error = output - expected_output

            gradient = self.__calculate_gradient(error, 
                                                 stimulus,
                                                 activation)
            
            
            

    def __test(self,):
        return None