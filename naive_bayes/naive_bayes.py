import numpy as np

from sklearn.naive_bayes import GaussianNB
from common.import_data import ImportData
from sklearn.model_selection import KFold


if __name__ == "__main__":
    data_set = ImportData()

    kf = KFold(n_splits=5, shuffle=True)
    # x = data_set.import_columns(['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
    #                             'Marginal Adhesion', 'Single Epithelial Cell Size','Bare Nuclei',
    #                             'Bland Chromatin', 'Normal Nucleoli'])
    x = data_set.import_all_data()
    y = data_set.import_columns(np.array(['Class']))
    print(x)
    scores = []
    for i in range(5):
        result = next(kf.split(x), None)
        x_train = x[result[0]]
        x_test = x[result[1]]
        y_train = y[result[0]]
        y_test = y[result[1]]
        NB = GaussianNB(var_smoothing=1e-9)
        NB.fit(x_test, y_test.ravel())
        y_predict = NB.predict(x_train)
        predictions = NB.predict(x_test)
        scores.append(NB.score(x_test, y_test))
        print('Scores from each Iteration: ', scores)

    print('Average K-Fold Score :', np.mean(scores))

