import numpy as np

from ANN.function import sigmoid


# -----------------------------------------------------------------------------------------------
# Création du concept de neurone, ici un neurone continu (!= seuil d'activation)
# -----------------------------------------------------------------------------------------------

class ContinuousNeuron:
    def __init__(self, number_of_inputs: int):
        self.weights = np.random.randn(number_of_inputs)
        self.bias = np.random.randn()

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        return sigmoid(weighted_sum)

    def update_weights(self, inputs: np.ndarray, learning_rate: float, delta: np.ndarray) -> None:
        self.weights += learning_rate * np.dot(inputs.T, delta)
        self.bias += learning_rate * np.sum(delta, axis=0)

