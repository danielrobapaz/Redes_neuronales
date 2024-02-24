import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

def sigmoid(z: np.array) -> np.array:
    return 1 / (1+np.exp(-z))

def sigmoid_derivate(z: np.array) -> np.array:
    s = sigmoid(z)
    return s * (s - 1)

class Neural_Network:
    def __init__(self,
                 data: pd.DataFrame,
                 number_of_neurons_hidden_layers: int = 0,
                 learning_rate: float = 0.01,
                 momentum: float = 0.0,
                 max_epoch: int = 100) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_epoch = max_epoch
        
        np.random.seed()
        train, test = train_test_split(data, train_size=0.8)

        # init weights
        input_size = len(data.columns) - 1
        self.weights_hidden_layer = np.random.randn(input_size,
                                                    number_of_neurons_hidden_layers)
        self.weights_output_layer = np.random.randn(number_of_neurons_hidden_layers,
                                                    1)
        
        self.bias_hidden_layer = np.random.randn(1, number_of_neurons_hidden_layers)
        self.bias_output_layer = np.random.randn(1, 1)

        self.prev_weights_hidden_layer = np.zeros((input_size, number_of_neurons_hidden_layers))
        self.prev_weights_output_layer = np.zeros((number_of_neurons_hidden_layers, 1))

        self.__train(train)
        self.__plot_error()
        self.__plot_data(data)
        self.__test(test)
        self.__show_metrics()

    def __train(self, data: pd.DataFrame) -> None:
        X = np.array(data[data.columns[:-1]])
        y = np.array(data[data.columns[-1]])
        y = np.reshape(y, (len(y), 1))

        mse_per_epoch = []
        for epoch in range(self.max_epoch):
            # feed forward
            hidden_layer = X.dot(self.weights_hidden_layer) + self.bias_hidden_layer
            activation_hidden_layer = sigmoid(hidden_layer)

            output_layer = activation_hidden_layer.dot(self.weights_output_layer) + self.bias_output_layer
            activation_output_layer = sigmoid(output_layer)

            # backpropagation
            error = activation_output_layer - y
            dw_output_layer = activation_hidden_layer.T.dot(error * sigmoid_derivate(output_layer))
            d_bias_output_layer = np.sum(error * sigmoid_derivate(output_layer),
                                         axis=0,
                                         keepdims=True)
            
            error_hidden_layer = error.dot(self.weights_output_layer.T) * sigmoid_derivate(hidden_layer)

            dw_hidden_layer = X.T.dot(error_hidden_layer)
            d_bias_hidden_layer = np.sum(error_hidden_layer,
                                         axis=0,
                                         keepdims=True)
            
            # update weights
            self.weights_hidden_layer += (self.learning_rate*dw_hidden_layer + self.momentum*self.prev_weights_hidden_layer)
            self.bias_hidden_layer += self.learning_rate*d_bias_hidden_layer
            self.weights_output_layer += (self.learning_rate*dw_output_layer + self.momentum*self.prev_weights_output_layer)
            self.bias_output_layer += self.learning_rate*d_bias_output_layer

            # update momentum
            self.prev_weights_hidden_layer = (self.learning_rate*dw_hidden_layer + self.momentum*self.prev_weights_hidden_layer).copy()
            self.prev_weights_output_layer = (self.learning_rate*dw_output_layer + self.momentum*self.prev_weights_output_layer).copy()

            mse = np.mean(error * error)
            mse_per_epoch.append(mse)
        
            if epoch % 50 == 0:
                print(f'Epoch {epoch} - MSE {mse}')

        self.error_per_epoch = mse_per_epoch        


    def predict(self, X: np.array) -> None:
        hidden_layer = X.dot(self.weights_hidden_layer) + self.bias_hidden_layer
        activation_hidden_layer = sigmoid(hidden_layer)

        output_layer = activation_hidden_layer.dot(self.weights_output_layer) + self.bias_output_layer
        activation_output_layer = sigmoid(output_layer)

        output = activation_output_layer[0][0]

        return 1 if output >= 0.3 else 0
        

    def __plot_data(self, data: pd.DataFrame) -> None:
        network_y = np.array([])
        data_columns = data.columns[:-1]
        
        for _, row in data.iterrows():
            x = np.array(row[data_columns])
            output = self.predict(x)

            network_y = np.append(network_y, output)

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


        _, ax = plt.subplots()
        ax.plot(x, self.error_per_epoch)

        ax.set_title('MSE por epoca')
        ax.set_xlabel('Epoca')
        ax.set_ylabel('MSE')
        plt.show()

    def __test(self, data: pd.DataFrame) -> None:
        tp = 0
        fp = 0
        fn = 0
        tn = 0


        data_columns = data.columns[:-1]
        target_column = data.columns[-1]

        for _, row in data.iterrows():
            x = np.array(row[data_columns])
            y = np.array(row[target_column].astype('int'))

            output = self.predict(x)

            result = (output, y)

            if result == (1,1):
                tp += 1

            elif result == (1, 0):
                fp += 1

            elif result == (0, 0):
                tn += 1

            else:
                fn += 1

        
        confussion_matrix = {
            'Positivo': [tp, fn],
            'Negativo': [fp, tn]
        }

        confussion_matrix = pd.DataFrame(confussion_matrix)
        confussion_matrix.index = ['Positivo', 'Negtivo']

        _, ax = plt.subplots()
        
        sns.heatmap(confussion_matrix, 
                    annot=True,
                    cmap=sns.color_palette('mako', as_cmap=True),
                    fmt=',d', 
                    ax=ax)
        
        ax.set_title('Matriz de confusion')
        ax.set_xlabel('Valor real')
        ax.set_ylabel('Prediccion')
        
        plt.show()


        self.tp = tp
        self.fp = fp
        self.tn = tn
        self.fn = fn

    def __show_metrics(self):
        tp = self.tp
        tn = self.tn
        fp = self.fp
        fn = self.fn
        
        # calculo de metricas
        accuracy = round((tp + tn) / (tp + fp + tn + fn), 3)
        precision = round(tp / (tp + fp), 3)
        sensivity = round(tp / (tp + fn), 3)
        especificity = round(tn / (tn + fp), 3)
        negative_predictive_value = round(tn / (tn + fn), 3)


        print(f'accuracy - {accuracy}')
        print(f'precision - {precision}')
        print(f'sensivity - {sensivity}')
        print(f'especificity - {especificity}')
        print(f'negative_predictive_value - {negative_predictive_value}')

