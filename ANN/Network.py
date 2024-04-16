import numpy as np

from ANN.Function import sigmoid, softmax


class Network:
    def __init__(self, sizes, epochs, lr):
        self.sizes = sizes
        self.epochs = epochs
        self.lr = lr

        # nombre de couches
        num_layers = len(sizes)

        # initialisation des paramètres
        self.params = {}
        for i in range(1, num_layers):
            self.params['W' + str(i)] = np.random.randn(sizes[i], sizes[i - 1]) * np.sqrt(1. / sizes[i])


    def forward(self, input):
        # W : weights, Z : weighted inputs, Y : outputs
        self.params['Y0'] = input
        num_layers = len(self.sizes)

        # Boucle sur les hidden et output layers (0=input layer; num_layers -1 = output layer)
        for i in range(1, num_layers):
            self.params['Z' + str(i)] = np.dot(self.params['W' + str(i)], self.params['Y' + str(i - 1)])
            if i != num_layers - 1:
                self.params['Y' + str(i)] = sigmoid(self.params['Z' + str(i)])
            else:
                self.params['Y' + str(i)] = softmax(self.params['Z' + str(i)])

        return self.params['Y' + str(num_layers - 1)]


    def backward(self, targets, output):
        #Attention nécessité de formater targets sur la base des labels
        params = self.params
        new_W = {}

        num_layers = len(self.sizes)

        # Calcul de l'erreur en sortie (MSE) et dérivée de l'erreur sur l'output layer
        error = 2 * (output - targets) / output.shape[0] * softmax(params['Z' + str(num_layers - 1)], derivative=True)

        for i in range(num_layers - 1, 1, -1):
            #Calcul des poids à actualiser(gradient)
            new_W['W' + str(i)] = np.outer(error, params['Y' + str(i - 1)])
            #Retropropagation de l'erreur
            error = np.dot(params['W' + str(i)].T, error) * sigmoid(params['Z' + str(i - 1)], derivative=True)

        # Calcul de l'actualisation(gradient) du premier hidden layer
        new_W['W1'] = np.outer(error, params['Y0'])

        return new_W


    def update(self, new_W):
        for key, value in new_W.items():
            #actualisation selon le produit learning rate * gradient
            self.params[key] -= self.lr * value


    def train(self, inputs, labels):
        print('Starting training...')
        expectations = self.labels_to_expectations(labels)

        for iteration in range(self.epochs):
            good_prediction = 0
            print('Epoch ', iteration, ' in progress')
            for input, expectation in zip(inputs, expectations):
                outputs = self.forward(input)
                new_W = self.backward(expectation, outputs)
                self.update(new_W)
                if np.argmax(outputs) == np.argmax(expectation):
                    good_prediction+= 1
            print('Epoch ', iteration,' done with accuracy ', round(good_prediction/len(inputs)*100,4),'%')




    def labels_to_expectations(self, labels):
        # Formate les labels (2) en tableau ([0,0,1,0,0,0,0,0,0,0]) pour le calcul des erreurs
        expectations = []
        for label in labels:
            expectation = np.zeros(10)
            expectation[label] = 1
            expectations.append(expectation)
        return expectations