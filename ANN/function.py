import numpy as np

# -----------------------------------------------------------------------------------------------
# Création des fonctions d'activation ici softmax et sigmoid
# -----------------------------------------------------------------------------------------------

def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
