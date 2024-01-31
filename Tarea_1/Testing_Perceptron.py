import pandas as pd
from Perceptron import Perceptron
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split

class Testing_Perceptron:
    def __init__(self, n: int, learning_rate: float = 1):
        df = pd.read_csv(f'datasets/{n}_3d_coords.csv')

        train, test = train_test_split(df, train_size=0.8, test_size=0.2)


        print(f'Testing with {n} examples')
        print(f'...:::Perceptron:::...')
        perceptron = Perceptron(train_data=train,
                                test_data=test,
                                algorithm = 'Perceptron',
                                learning_rate = learning_rate)
        
        print('\n\n...:::Averaged Perceptron:::...')
        averaged_perceptron = Perceptron(train_data=train,
                                         test_data=test,
                                         algorithm = 'Averaged',
                                         learning_rate = learning_rate)
        
        print('\n\n...:::MIRA Perceptron:::...')
        mira_perceptron = Perceptron(train_data=train,
                                     test_data=test,
                                     algorithm = 'MIRA',
                                     learning_rate = learning_rate)
        

        fig, ax = plt.subplots()
        x = np.linspace(1, len(averaged_perceptron.error_per_epoch), len(averaged_perceptron.error_per_epoch))
        ax.plot(x, averaged_perceptron.error_per_epoch, label = 'Averaged', linestyle = '--')
        
        x = np.linspace(1, len(perceptron.error_per_epoch), len(perceptron.error_per_epoch))
        ax.plot(x, perceptron.error_per_epoch, label = 'Perceptron')

        x = np.linspace(1, len(mira_perceptron.error_per_epoch), len(mira_perceptron.error_per_epoch))
        ax.plot(x, mira_perceptron.error_per_epoch, label = 'MIRA', linestyle = ':')

        ax.set_title(f'Cantidad de ejemplos mal clasificados por epoca. {n} ejemplos')
        ax.set_xlabel('Epoca')
        ax.set_ylabel('Cantidad de ejemplos mal clasificados')
        ax.legend()

        fig.set_figwidth(12)

        plt.show()

tp = Testing_Perceptron(10000, 0.7)