import matplotlib.pyplot as plt

def plot_mnist_images_range(data, labels, start_index, end_index):
    # Vérifier que les indices soient valides
    if start_index < 0:
        start_index = 0
    if end_index >= len(data):
        end_index = len(data) - 1

    num_images = end_index - start_index + 1

    # Créer une grille de sous-plots pour afficher les images
    fig, axes = plt.subplots(1, num_images, figsize=(10, 2))

    for i in range(num_images):
        # Obtenir l'image et l'étiquette correspondante
        image = data[start_index + i].reshape(28, 28)  # Les images MNIST sont de taille 28x28
        label = labels[start_index + i]

        # Afficher l'image
        axes[i].imshow(image, cmap='gray')
        axes[i].set_title(f"Label: {label}")
        axes[i].axis('off')  # Masquer les axes

    plt.tight_layout()
    plt.show()


