import pandas as pd
import numpy as np
from math import exp
from sklearn.model_selection import train_test_split

class Neuron:
    def __init__(self, 
                 number_of_weights: int):
        self.weigths = np.random.random(number_of_weights + 1)
        self.input = None

    def activate(self, input_data: pd.DataFrame) -> float:
        self.input = np.array(input_data)
        stimulus = np.dot(input_data, self.weigths)
        return self.__sigmoid(stimulus), stimulus
    
    def __sigmoid(self, value: float) -> float:
        return 1/(1+exp(-value))
    
    def activate_derivate(self, value: float) -> float:
        return self.__sigmoid(value)*(1-self.__sigmoid(value))


class Neural_Network:
    def __init__(self,
                 data: pd.DataFrame,
                 layers: [int] = [1],
                 learning_rate: float = 0.01):
        
        if not layers[-1] == 1:
            raise Exception('Unexpected number of neurons in output layer')
        
        variables = len(data.columns)-1
        self.layers = []
        for i, number_of_neurons in enumerate(layers):
            if i == 0:
                self.layers.append([Neuron(variables) for _ in range(number_of_neurons)])

            else:
                self.layers.append([Neuron(layers[i-1]) for _ in range(number_of_neurons)])

        self.answer_map = [[1, 0], [0, 1]]

        self.learning_rate = learning_rate
        train, test = train_test_split(data, train_size=0.8, test_size=0.2)
        self.__train(train)
        self.__test(test)

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
                    neuron_gradient = -error*derivate*activation[i][j]

                else:
                    # hidden layer
                    next_layer_weights = np.array([n.weigths[j] for n in self.layers[i+1]])
                    neuron_gradient = derivate*np.dot(next_layer_weights, gradient[0])
                    
                current_gradient.append(neuron_gradient)
            
            gradient.insert(0, np.array(current_gradient))

        return gradient

    def __update_neurons(self, gradient: [np.array]):
        for i in range(len(self.layers)):
            for j in range(len(self.layers[i])):
                y = self.layers[i][j].input
                self.layers[i][j].weigths = self.layers[i][j].weigths + self.learning_rate*gradient[i][j]*y

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
                
                activation, stimulus = self.__feed_fordward(x)

                raw_output = activation[-1]
                output = 1 if raw_output >= 0.5 else -1
                error = d-output

                if output == d:
                    bad_clasified_examples += 1 
                gradient = self.__calculate_gradient(error, 
                                                    stimulus,
                                                    activation)
                self.__update_neurons(gradient)

            bad_clasified_examples_per_epoch.append(bad_clasified_examples)

            percentage_error = bad_clasified_examples/len(data)
            if percentage_error <= 0.01 or current_epoch >= 1000:
                print(f'Convergence reached in {current_epoch} epochs with {percentage_error}')
                print(bad_clasified_examples_per_epoch[980:])
                break

            print(current_epoch)

    def __test(self, data: pd.DataFrame):
        return None