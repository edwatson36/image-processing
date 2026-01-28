import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.utils import load_img
from pathlib import Path

# Plot given images
def plot_sample_images(images_path: list[Path], target_size: tuple[int, int] | None = None) -> None:
    """
    Plots a list of image paths in a single row.

    Args:
        image_paths (list[Path]): List of paths to images.
    """
    if not image_paths:
        raise ValueError("No image paths provided to plot.")
    
    plt.figure(figsize=(15, 6))
    for i, img_path in enumerate(image_paths):
        img = load_img(img_path, target_size=target_size)
        plt.subplot(1, len(image_paths), i + 1)
        plt.imshow(img)
        plt.title(img_path.parent.name, fontsize=10)
        plt.axis("off")
    plt.tight_layout()
    plt.show()


# Visualise one batch
def show_batch(dataset: tf.data.Dataset, class_names: list[str] = None, n: int = 9) -> None:
    """
    Display a grid of sample images from a TensorFlow dataset.

    Args:
        dataset (tf.data.Dataset): Dataset yielding (image, label) pairs.
        class_names (list[str], optional): List mapping class IDs to names.
        n (int): Number of images to display (default 9).

    Returns:
        None — displays the image grid using matplotlib.
    """
    plt.figure(figsize=(10, 10))
    # Take one batch from the dataset
    for images, labels in dataset.take(1):
        for i in range(min(n, len(images))):
            ax = plt.subplot(int(n**0.5), int(n**0.5), i + 1)
            plt.imshow(images[i].numpy())
            label = int(labels[i].numpy())
            if class_names:
                plt.title(class_names[label])
            else:
                plt.title(f"Class: {label}")
            plt.axis("off")
    plt.tight_layout()
    plt.show()
