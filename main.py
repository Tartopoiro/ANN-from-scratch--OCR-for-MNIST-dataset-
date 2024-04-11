from loader import load_mnist_data
from printer import plot_mnist_images_range

train_data, train_labels, test_data, test_labels = load_mnist_data('mnist_train.csv', 'mnist_test.csv')
plot_mnist_images_range(train_data, train_labels, start_index=5, end_index=9)