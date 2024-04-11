import numpy as np
from typing import List

from ANN.layer import Layer


# -----------------------------------------------------------------------------------------------
# Création d'une classe NeuralNetwork
# -----------------------------------------------------------------------------------------------

class NeuralNetwork:
    def __init__(self, number_of_inputs: int, layer_dimensions: List[int]):
        # Initialisation du layer 1
        self.layers = [Layer(layer_dimensions[0], None, None, number_of_inputs)]
        # ajout des layers input
        for i in range(len(layer_dimensions) - 1):
            self.layers.append(Layer(layer_dimensions[i + 1], None, self.layers[i]))
        # ajout output
        for i in range(len(self.layers) - 1):
            if i < len(self.layers):
                self.layers[i].output_layer = self.layers[i + 1]

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs

    def print_layers(self):
        for i, layer in enumerate(self.layers):
            print(f"Layer {i + 1} - Dimension: {layer.dimension}")

            if layer.input_layer:
                print(f"  Input from Layer {i} with Dimension {layer.input_layer.dimension}")

            if layer.output_layer:
                print(f"  Output to Layer {i + 2} with Dimension {layer.output_layer.dimension}")

            print("\nNeurons:")
            for j, neuron in enumerate(layer.neurons):
                print(f"  Neuron {j + 1} - Weights: {neuron.weights}, Bias: {neuron.bias}")

            print("\n------------------------")

    def calculate_loss(self, outputs: np.ndarray, expected_outputs: np.ndarray) -> float:
        return 0.5 * np.sum((outputs - expected_outputs) ** 2)

    def train(self, inputs: np.ndarray, expected_outputs: np.ndarray, learning_rate: float, epochs: int) -> None:
        for epoch in range(epochs):
            for i, row in enumerate(inputs):
                outputs = self.forward(inputs[i])
                loss = self.calculate_loss(outputs, expected_outputs[i])
                print("In :", inputs[i], " Out: ", outputs, " Real Out: ", expected_outputs[i], " loss: ", loss)


            #deltas = [self.layers[-1].calculate_deltas(expected_outputs - outputs)]
            #for layer in reversed(self.layers[:-1]):
                #deltas.append(layer.calculate_deltas(np.dot(deltas[-1], layer.output_layer.neurons[0].weights.T)))
                #deltas.reverse()

                #for layer, delta in zip(self.layers, deltas):
                #    layer.update_weights(inputs, learning_rate, delta)


# -----------------------------------------------------------------------------------------------
# TEST -- TEST -- TEST -- TEST -- TEST -- TEST -- TEST -- TEST -- TEST -- TEST -- TEST -- TEST
# -----------------------------------------------------------------------------------------------





inputs = np.array([[1.0, 2.0, 1.0, 6.0], [2.0, 3.0, 3.0, 1.0], [2.0, 4.0, 2.0, 5.0], [4.0, 5.0, 2.0, 5.0], [4.0, 5.0, 2.0, 5.0]])
nn = NeuralNetwork(inputs.shape[1], [2, 4, 6, 6, 4, 2, 1])
#nn.print_layers()
expected_outputs = np.array([1.0, 3.0, 2.0, 5.0, 5.0])
learning_rate = 0.01
epochs = 10
nn.train(inputs, expected_outputs, learning_rate, epochs)


print(nn.forward(inputs[1]))
