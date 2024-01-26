import pandas as pd
import numpy as np

class Dataset_generator():

    def __generate_dataset(self, n: int) -> pd.DataFrame:
        df = pd.DataFrame()
        df['x'] = 10*np.random.random(n)
        df['y'] = 10*np.random.rand(n)
        df['z'] = 10*np.random.rand(n)
        
        return df

    def __save_dataset(self, df: pd.DataFrame, 
                       route: str):
        df.to_csv(f'datasets/{route}.csv', index=False)

    def gen(self):
        number_elements = [100, 1000, 10000]

        for number in number_elements:
            self.__save_dataset(df=self.__generate_dataset(number),
                                route=f'{number}_3d_coords')