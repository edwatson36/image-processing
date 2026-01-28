import tensorflow as tf
import pandas as pd
from pathlib import Path
from constants import CLASS_FLAG, IMAGE_PATH

# Preprocess image and encode class label
def preprocess_image(
    image_path: str,
    label: int,
    num_classes: int,
    image_size: tuple[int, int] = (224, 224),
    inc_res_v2: bool = True,
    one_hot: bool = True
) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Load and preprocess a single image: decode, resize, normalize.
    Encode label if one_hot=True.

    Args:
        image_path (str): Path to the image file.
        label (int): Class label.
        num_classes (int): Total number of classes in the dataset
        image_size (tuple[int, int]): Target image size.
        inc_res_v2 (bool): Normalise pixels to [-1,1] range if inc_res_v2 True else, [0,1]
        one_hot (bool): Whether to one-hot encode the label.

    Returns:
        (image, label): Tuple of preprocessed image tensor and label.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    # Normalise pixels
    if inc_res_v2:
        image = tf.keras.applications.inception_resnet_v2.preprocess_input(image)
    else:
        image = image / 255.0  # Normalize [0,1]

    if one_hot:
        label = tf.one_hot(label, depth=num_classes)

    return image, label


# Convert a DataFrame to a tf.data.Dataset
def df_to_dataset(
    df: pd.DataFrame,
    base_dir: Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    inc_res_v2: bool = True,
    one_hot: bool = True,
    shuffle: bool = True
) -> tf.data.Dataset:
    """
    Convert a DataFrame into a TensorFlow Dataset.

    Args:
        df (pd.DataFrame): DataFrame containing a Path to each image and class id flag.
        base_dir (Path): Base directory where images are stored.
        image_size (tuple[int, int]): Target image size.
        batch_size (int): Batch size for dataset.
        shuffle (bool): Whether to shuffle dataset.

    Returns:
        tf.data.Dataset: TensorFlow dataset with (image, label) pairs.
    """
    image_paths = [str(base_dir / p) for p in df[IMAGE_PATH]]
    labels = df[CLASS_FLAG].astype("int32").values
    num_classes = df[CLASS_FLAG].nunique()

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(lambda x, y: preprocess_image(x, y, num_classes, image_size, inc_res_v2, one_hot),
                num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df))
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


# Final wrapper function
def create_tf_datasets(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    base_dir: Path,
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    inc_res_v2: bool = True,
    one_hot: bool = True
) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
    """
    Create TensorFlow datasets for training, validation, and testing.

    Args:
        train_df, val_df, test_df (pd.DataFrame): DataFrame containing a Path to each image and class id flag.
        base_dir (Path): Path to dataset root folder.
        image_size (tuple[int, int]): Target image size.
        batch_size (int): Batch size for dataset.
        inc_res_v2 (bool): Normalise pixels to [-1,1] range if inc_res_v2 True else, [0,1]
        one_hot (bool): Whether to one-hot encode the label.

    Returns:
        (train_ds, val_ds, test_ds): Ready-to-use TensorFlow Datasets.
    """
    train_ds = df_to_dataset(train_df, base_dir, image_size, batch_size, inc_res_v2, one_hot, shuffle=True)
    val_ds = df_to_dataset(val_df, base_dir, image_size, batch_size, inc_res_v2, one_hot, shuffle=False)
    test_ds = df_to_dataset(test_df, base_dir, image_size, batch_size, inc_res_v2, one_hot, shuffle=False)

    return train_ds, val_ds, test_ds
