import numpy as np

from sklearn.model_selection import train_test_split
from common.import_data import ImportData
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier


if __name__ == "__main__":
    data_set = ImportData()

    x_train, x_test, y_train, y_test = train_test_split(data_set.import_all_data(),
                                                        data_set.import_columns(np.array(['Class'])),
                                                        test_size=0.2, random_state=0)

    random_forest = RandomForestClassifier(n_estimators=100)
    random_forest.fit(x_train, y_train.ravel())
    y_prediction = random_forest.predict(x_test)
    print(y_test)
    print(y_prediction)
    print(accuracy_score(y_test, y_prediction.ravel()))
