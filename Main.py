from ANN.Network import Network
from IO.FileLoader import load_mnist_data
from IO.Printer import plot_mnist_images_range

# All the code are using english naming but comment inside files are in french, I will translate it (one day ^^)
# Feel free to contact me at bbasset.benjamin@gmail.com for more explanation.


# Extracting data from CSV using the function in IO\FileLoader
train_data, train_labels, test_data, test_labels = load_mnist_data('mnist_train.csv', 'mnist_test.csv')

# Now you can create the ANN you want (here 2 hidden layers)
# => Activation function are not choosable
# => softmax for output layer and sigmoid for others
# CAUTION : No batch so that will take time ^^
#
ann = Network(sizes=[len(train_data[0]),256, 128, 64, 10], epochs=8, lr=0.01)
ann.train(train_data, train_labels)