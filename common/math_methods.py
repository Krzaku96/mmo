import numpy as np


class MathFunctions:

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def transform_into_discrete_values(x: np.ndarray) -> np.ndarray:
        return x / 2 - 1

    @staticmethod
    def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
        return x * (1.0 - x)
