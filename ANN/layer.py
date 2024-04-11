from typing import List, Optional

import numpy as np

from neuron import ContinuousNeuron


# -----------------------------------------------------------------------------------------------
# Création du concept de couche de neurone
# -----------------------------------------------------------------------------------------------
class Layer:
    def __init__(self, dimension: int, output_layer: Optional['Layer'] = None, input_layer: Optional['Layer'] = None,
                 number_of_input: Optional[int] = 0):
        self.dimension = dimension
        self.output_layer: Optional['Layer'] = output_layer
        self.input_layer: Optional['Layer'] = input_layer
        self.number_of_input = number_of_input
        if input_layer:
            self.number_of_input = input_layer.dimension
        self.neurons = [ContinuousNeuron(self.number_of_input) for _ in range(dimension)]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return np.array([neuron.forward(inputs) for neuron in self.neurons])

    def update_weights(self, inputs: np.ndarray, learning_rate: float) -> None:
        for neuron in self.neurons:
            neuron.update_weights(inputs, learning_rate)

