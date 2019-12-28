import numpy as np

from sklearn.neural_network import MLPClassifier
from common.import_data import ImportData
from sklearn.model_selection import KFold

if __name__ == "__main__":
    data_set = ImportData()

    kf = KFold(n_splits=5, shuffle=True)
    #x = data_set.import_columns(['Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
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
        NN = MLPClassifier(solver='adam', alpha=0.5151111,
                           hidden_layer_sizes=(20, 1),
                           random_state=5, max_iter=2000, verbose=1).fit(x_train, y_train.ravel())
        predictions = NN.predict(x_test)
        scores.append(NN.score(x_test, y_test))
        print('Scores from each Iteration: ', scores)

    print('Average K-Fold Score :', np.mean(scores))


