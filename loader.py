import csv
import numpy as np
#Charge l'ensemble du dataset, entrainement et test
def load_mnist_data(train_file, test_file):
    # Charger les données d'entraînement
    train_data = []
    train_labels = []

    with open(train_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)  # Ignorer la première ligne (en-tête)
        for row in csv_reader:
            label = int(row[0])
            # normalisation : passage d'une base 256 à une base 0 à 1
            image_data = np.array([int(x) / 255.0 for x in row[1:]], dtype=np.float32)
            train_data.append(image_data)
            train_labels.append(label)

    # Charger les données de test (similaire à l'entraînement)
    test_data = []
    test_labels = []

    with open(test_file, 'r') as file:
        csv_reader = csv.reader(file)
        next(csv_reader)
        for row in csv_reader:
            label = int(row[0])
            image_data = np.array([int(x) / 255.0 for x in row[1:]], dtype=np.float32)
            test_data.append(image_data)
            test_labels.append(label)

    # Convertir les listes en tableaux NumPy pour une manipulation plus facile
    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    return train_data, train_labels, test_data, test_labels
