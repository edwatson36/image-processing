from pathlib import Path
import os
import tarfile
from urllib.request import urlretrieve
import pandas as pd
import random
from .constants import CORE_METADATA_FILES, COLUMNS

# Download data function
def download_dataset(url: str = DATA_URL, save_path: Path = ARCHIVE_PATH) -> Path:
    """Download dataset archive if not already present."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if not save_path.exists():
        print(f"Downloading dataset from {url}...")
        urlretrieve(url, save_path)
        print("Download complete.")
    else:
        print("Archive already exists.")
    return save_path


# Extract data function
def extract_dataset(archive_path: Path = ARCHIVE_PATH, extract_to: Path = EXTRACT_DIR) -> Path:
    """Extract dataset from tar archive."""
    if not extract_to.exists():
        print(f"Extracting {archive_path}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_to.parent)
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")
    return extract_to


# Fetch train-test split
def fetch_split(dataset_dir: Path) -> pd.DataFrame:
    """
    Fetch the CUB_200_2011 train-test split file 'train_test_split.txt' and return as df.

    Args:
        dataset_dir (Path): Path to the folder containing the split file.

    Returns:
        pd.DataFrame: DataFrame with columns ['image_id', 'is_training'].
    """
    split_path = dataset_dir / 'train_test_split.txt'
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    split_df = pd.read_csv(
        split_path,
        sep=" ",
        header=None,
        names=["image_id", "is_training"]
    )
    print(f"Loaded split file 'train_test_split.txt' with {len(split_df)} entries")
    return split_df


# Fetch core metadata files (image_id, class)
def fetch_core_metadata(dataset_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch the CUB_200_2011 files containing image_id, image_path, class_id, class_name and return as df.
    Files fetched: images.txt, image_class_labels.txt, classes.txt

    Args:
        dataset_dir (Path): Path to the folder containing the metadata files.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (images_df, image_class_labels_df, classes_df)
    """
    # File names
    filename_list = ["images.txt", "image_class_labels.txt", "classes.txt"]

    # Check file paths
    for filename in filename_list:
        file_path = dataset_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

    # Load core metadata files
    images_df = pd.read_csv(dataset_dir / filename_list[0], sep = " ", header=None, names=["image_id", "image_path"])
    image_class_labels_df = pd.read_csv(dataset_dir / filename_list[1], sep = " ", header=None, names=["image_id", "class_id"])
    classes_df = pd.read_csv(dataset_dir / filename_list[2], sep = " ", header=None, names=["class_id", "class_name"])

    print(f"Loaded {filename_list[0]}, {filename_list[1]}, {filename_list[2]} with {len(images_df)}, {len(image_class_labels_df)}, and {len(classes_df)} entries respectively")
    return images_df, image_class_labels_df, classes_df


# Merge metadata
def merge_metadata(dataset_dir: Path) -> pd.DataFrame:
    """
    Merge the core metadata files and train-test split into a single DataFrame.
    
    Args:
        dataset_dir (Path): Path to the folder containing the metadata and train-test split files.

    Returns:
        pd.DataFrame: 
            DataFrame with columns ['image_id', 'image_path', 'class_id', 'class_name', 'is_training'].
    """
    # Fetch core metadata
    images_df, image_class_labels_df, classes_df = fetch_core_metadata(dataset_dir)

    # Fetch train-test split
    split_df = fetch_split(dataset_dir)

    # Merge
    merged_df = (
        images_df
        .merge(image_class_labels_df, on="image_id")
        .merge(classes_df, on="class_id")
        .merge(split_df, on="image_id")
    )

    print(
        f"Merged metadata: {len(merged_df)} rows, "
        f"{len(merged_df.columns)} columns -> {list(merged_df.columns)}"
    )
    return merged_df


# Fetch random sample images
def fetch_sample_images(image_dir: Path, n: int = 5) -> list[Path]:
    """
    Randomly select one species folder and return a list of 'n' sample image paths.

    Args:
        image_dir (Path): Directory containing all species subfolders.
        n (int): Number of images to sample from the selected species.

    Returns:
        list[Path]: List of image file paths.
    """
    # Create a list of species folder paths with list comprehension
    species_folders = sorted([p for p in image_dir.iterdir() if p.is_dir()])
    # .iterdir() yields all immediate children of a directory as Path objects
    # .is_dir() yields True if entry is a directory, False if file
    if not species_folders:
        raise ValueError(f"No species folders found in {image_dir}.")

    # Choose one random species folder
    sample_species = random.choice(species_folders)
    print(f"Randomly selected species: {sample_species.name}")

    # Create a list of up to 'n' image paths
    image_paths = list(sample_species.glob("*.jpg"))[:n]  # first n images
    # .glob() yields all files that match the "*.jpg" pattern as Path objects
    if not image_paths:
        raise ValueError(f"No images found in {sample_species}.")

    return image_paths
