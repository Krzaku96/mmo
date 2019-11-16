from typing import Tuple

import numpy as np

from common.import_data import ImportData
from src.shared.math_functions import MathFunctions


class NeuralNetwork:
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.input = x
        self.weights1 = np.random.rand(self.input.shape[1], 4)
        self.weights2 = np.random.rand(4, 1)
        self.y = y
        self.output = np.zeros(self.y.shape)

    def feed_forward(self):
        self.layer = MathFunctions.sigmoid(np.dot(self.input, self.weights1))
        self.output = MathFunctions.sigmoid(np.dot(self.layer, self.weights2))

    def back_propagation(self):
        # derivative of the loss function - layer to output
        d_w2 = np.dot(self.layer.T, (2 * (self.y - self.output) * MathFunctions.sigmoid_derivative(self.output)))
        d_w1 = np.dot(self.input.T, (np.dot(2 * (self.y - self.output)
                                            * MathFunctions.sigmoid_derivative(self.output), self.weights2.T)
                                     * MathFunctions.sigmoid_derivative(self.layer)))
        self.update_weights(d_w1, d_w2)

    def update_weights(self, d_w1, d_w2):
        self.weights1 += d_w1
        self.weights2 += d_w2

    def train_network(self, x_ndarray: np.ndarray, y_ndarray: np.ndarray):
        length = x_ndarray.shape[0]
        for i in range(300):
            for j in range(length):
                x_data_temp = np.array([x_ndarray[j]])
                y_data_temp = np.array([y_ndarray[j]])
                self.input = x_data_temp
                self.y = y_data_temp
                self.feed_forward()
                self.back_propagation()

    def predict_value(self, x_ndarray: np.ndarray, y_ndarray: np.ndarray) -> Tuple[float, float]:
        self.input = x_ndarray
        self.y = y_ndarray
        self.feed_forward()
        predicted_value = self.output[0, 0]
        true_value = self.y[0, 0]
        return predicted_value, true_value


if __name__ == "__main__":
    data_set = ImportData()
    x1: np.ndarray = data_set.import_all_data()
    y1: np.ndarray = data_set.import_columns(np.array(['Class']))

    y1 = MathFunctions.transform_into_discrete_values(y1)
    length = x1.shape[0]

    temp_x = np.array([x1[0]])
    temp_y = np.array([y1[0]])
    neural_network = NeuralNetwork(temp_x, temp_y)
    neural_network.train_network(x1, y1)

    right_values = 0
    values = 0

    for i in range(length):
        temp_x = np.array([x1[i]])
        temp_y = np.array([y1[i]])
        predicted_value, true_value = neural_network.predict_value(temp_x, temp_y)

        round_p = round(predicted_value)
        round_t = round(true_value)
        if round_p == round_t:
            right_values += 1
        values += 1
        print('Iteracja \t', i, 'Oryginalna wartość:\t', true_value, 'Przewidywana:\t', predicted_value)

    print('Rozpoznano poprawnie: ', right_values, ' na ', values)

