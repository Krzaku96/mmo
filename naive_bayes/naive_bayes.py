import numpy as np

from sklearn.naive_bayes import GaussianNB
from common.import_data import ImportData


if __name__ == "__main__":
    data_set = ImportData()
    x_train: np.ndarray = data_set.import_train_data_bayes()
    x_test: np.ndarray = data_set.import_test_data_bayes()
    y_train: np.ndarray = data_set.import_columns_train_bayes(np.array(['Class']))
    y_test: np.ndarray = data_set.import_columns_test_bayes(np.array(['Class']))
    NB = GaussianNB()
    NB.fit(x_test, y_test.ravel())
    y_predict = NB.predict(x_train)
    print("Skuteczność NB: {:.2f}".format(NB.score(x_train, y_train.ravel())))


