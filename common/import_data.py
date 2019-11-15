import pandas as pd
import numpy


class ImportData:
    def __init__(self,
                 dataset_path='../dataset/breast-cancer-wisconsin.data',
                 columns_path='../../data/breast-cancer-columns.names'):
        self.data_path = dataset_path
        self.columns_path = columns_path

    def import_names_of_columns(self)-> numpy.ndarray:
        columns = pd.read_csv(self.columns_path, setp=',', comment='#', header=None).to_numpy()

        return numpy.concatenate(columns, axis=0)

    def import_columns(self, selected_columns_names: numpy.ndarray)-> numpy.ndarray:

        columns_names = self.import_names_of_columns()

        data = pd.read_csv(self.dataset_path,
                           sep=',',
                           names=columns_names,
                           usecols=selected_columns_names)

        return data.values



