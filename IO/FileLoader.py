import csv
import numpy as np
#Charge l'ensemble du dataset, entrainement et test
def load_mnist_data(train_file, test_file):
    # Charge les données d'entraînement
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

    # Charge les données de test
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

    return train_data, train_labels, test_data, test_labels