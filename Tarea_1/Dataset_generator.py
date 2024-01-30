import pandas as pd
import numpy as np

MIN_VALUE = -10
MAX_VALUE = 10

class Dataset_generator():
    def __generate_dataset(self, n: int) -> pd.DataFrame:
        df = pd.DataFrame()
        df['x'] = (MAX_VALUE - MIN_VALUE + 1) * np.random.random_sample(n) + MIN_VALUE
        df['y'] = (MAX_VALUE - MIN_VALUE + 1) * np.random.random_sample(n) + MIN_VALUE
        df['z'] = (MAX_VALUE - MIN_VALUE + 1) * np.random.random_sample(n) + MIN_VALUE
        df['category'] = df['z'].apply(lambda z: 1 if z >= 0 else -1)
        
        return df

    def __save_dataset(self, df: pd.DataFrame, 
                       route: str):
        df.to_csv(f'datasets/{route}.csv', index=False)

    def gen(self):
        number_elements = [100, 1000, 10000]

        for number in number_elements:
            self.__save_dataset(df=self.__generate_dataset(number),
                                route=f'{number}_3d_coords')