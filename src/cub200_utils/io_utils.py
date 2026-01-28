from pathlib import Path
import os
import tarfile
from urllib.request import urlretrieve
import pandas as pd
import random
from typing import List, Dict, Tuple
from .constants import (
    DATA_URL,
    ARCHIVE_NAME,
    DATASET_NAME,
    CORE_METADATA_FILES,
    COLUMNS,
    JOIN_KEYS,
    JOIN_VALIDATION,
)

# Download data function
def download_dataset(data_dir: Path, url: str = DATA_URL, archive_name: str = ARCHIVE_NAME) -> Path:
    """Download dataset archive if not already present."""
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / archive_name
    if not archive_path.exists():
        print(f"Downloading dataset from {url}...")
        urlretrieve(url, archive_path)
        print("Download complete.")
    else:
        print("Archive already exists.")
    return archive_path


# Extract data function
def extract_dataset(archive_path: Path, extract_path: Path | None = None, dataset_name: str = DATASET_NAME) -> Path:
    """Extract dataset from tar archive."""
    # Define the extract_path, if given. Default to using the archive_path
    # For use if you want the extracts to be in a different location to the download
    if extract_path is None:
        extract_path = archive_path.parent

    dataset_path = extract_path / dataset_name

    if not dataset_path.exists():
        print(f"Extracting {archive_path}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(path=extract_path, filter="data")
        print("Extraction complete.")
    else:
        print("Dataset already extracted.")
    return dataset_path


# Fetch train-test split
def fetch_split(dataset_dir: Path) -> pd.DataFrame:
    """
    Fetch the train/test split metadata for the CUB_200_2011 dataset.

    This function fetches the split file using ``CORE_METADATA_FILES["split"]``
    and parses it using column definitions from ``COLUMNS["split"]``.

    Args:
        dataset_dir (Path): Root directory of the extracted CUB_200_2011 dataset.

    Returns:
        pd.DataFrame: DataFrame mapping image IDs to a training flag.

    Raises:
        FileNotFoundError: If the split metadata file cannot be found.
    """
    split_path = dataset_dir / CORE_METADATA_FILES["split"]
    if not split_path.exists():
        raise FileNotFoundError(f"Missing split file: {split_path}")

    split_df = pd.read_csv(
        split_path,
        sep=" ",
        header=None,
        names=COLUMNS["split"]
    )
    print(f"Loaded split file {CORE_METADATA_FILES["split"]} with {len(split_df)} entries")
    return split_df


# Fetch core metadata files (image_id, class)
def fetch_core_metadata(dataset_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetch the image_id, image_path, class_id, and class_name metadata for the CUB_200_2011 dataset.

    This function fetches the required metadata files using ''CORE_METADATA_FILES'' and
    parses them using column definitions from ''COLUMNS''.

    Args:
        dataset_dir (Path): Root directory of the extracted CUB_200_2011 dataset.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            (images_df, image_class_labels_df, classes_df)

    Raises:
        FileNotFoundError: If any of the required files cannot be found.
    """
    # File names
    filename_list = [CORE_METADATA_FILES["images"],
                     CORE_METADATA_FILES["image_class_labels"],
                     CORE_METADATA_FILES["classes"]]

    # Check file paths
    for filename in filename_list:
        file_path = dataset_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing file: {file_path}")

    # Load core metadata files
    images_df = pd.read_csv(dataset_dir / filename_list[0], sep = " ", header=None, names=COLUMNS["images"])
    image_class_labels_df = pd.read_csv(dataset_dir / filename_list[1], sep = " ", header=None, names=COLUMNS["image_class_labels"])
    classes_df = pd.read_csv(dataset_dir / filename_list[2], sep = " ", header=None, names=COLUMNS["classes"])

    print(f"Loaded {filename_list[0]}, {filename_list[1]}, {filename_list[2]} with {len(images_df)}, {len(image_class_labels_df)}, and {len(classes_df)} entries respectively")
    return images_df, image_class_labels_df, classes_df


# Merge metadata
def merge_metadata(dataset_dir: Path) -> pd.DataFrame:
    """
    Merge metadata fetched by fetch_core_metadata() with train-test split fetched by fetch_split().

    Args:
        dataset_dir (Path): Root directory of the extracted CUB_200_2011 dataset.

    Returns:
        pd.DataFrame:
            DataFrame with columns ['image_id', 'image_path', 'class_id', 'class_name', 'is_training'].

    Raises:
        FileNotFoundError: If any of the required files cannot be found.
        ValueError: If the output DataFrame is empty.

    """
    # NOTE:
    # This function could be further generalized by refactoring
    # fetch_core_metadata() to return a dict of DataFrames and driving
    # the merge logic via a declarative MERGE_PLAN in constants.py.

    # Fetch core metadata
    images_df, image_class_labels_df, classes_df = fetch_core_metadata(dataset_dir)

    # Fetch train-test split
    split_df = fetch_split(dataset_dir)

    # Merge
    merged_df = (
        images_df
        .merge(image_class_labels_df,
               on=JOIN_KEYS["image_labels"],
               validate=JOIN_VALIDATION["image_labels"])
        .merge(classes_df,
               on=JOIN_KEYS["class_labels"],
               validate=JOIN_VALIDATION["class_labels"])
        .merge(split_df,
               on=JOIN_KEYS["image_split"],
               validate=JOIN_VALIDATION["image_split"])
    )

    if merged_df.empty:
        raise ValueError(
            f"Merged metadata is empty. Check input files and join keys."
            f"Sizes: "
            f"images={len(images_df)}, "
            f"labels={len(image_class_labels_df)}, "
            f"classes={len(classes_df)}, "
            f"split={len(split_df)}"
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
