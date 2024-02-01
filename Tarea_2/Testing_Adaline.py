import pandas as pd
from LMS import LMS
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split

class Testing_Adaline:
    def __init__(self, n: int, learning_rate: float = 1):
        df = pd.read_csv(f'datasets/{n}_3d_coords_lms.csv')

        train, test = train_test_split(df, train_size=0.8, test_size=0.2)


        print(f'Testing with {n} examples')
        print(f'...:::LMS:::...')
        lms = LMS(train, test, learning_rate)

        fig, ax = plt.subplots()
        x = np.linspace(1, len(lms.error_per_epoch), len(lms.error_per_epoch))
        ax.plot(x, lms.error_per_epoch)

        ax.set_title(f'MSE por epoca. {n} ejemplos')
        ax.set_xlabel('Epoca')
        ax.set_ylabel('MSE')

        fig.set_figwidth(12)

        plt.show()

tp = Testing_Adaline(1000, 0.1)