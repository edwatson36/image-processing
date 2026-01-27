# Correct and working URL download as of 23/01/2026
DATA_URL = "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"

# Folder name
ARCHIVE_NAME = "CUB_200_2011.tgz"
DATASET_NAME = "CUB_200_2011"

# Intrinsic constants for CUB_200_2011 if downloaded from DATA_URL
CORE_METADATA_FILES = {
    "images": "images.txt",
    "image_class_labels": "image_class_labels.txt",
    "classes": "classes.txt",
    "split": "train_test_split.txt"
}

COLUMNS = {
    "images": ["image_id", "image_path"],
    "image_class_labels": ["image_id", "class_id"],
    "classes": ["class_id", "class_name"],
    "split": ["image_id", "is_training"]
}

JOIN_KEYS = {
    "image_labels": "image_id",
    "image_split": "image_id",
    "class_labels": "class_id",
}

JOIN_VALIDATION = {
    "image_labels": "one_to_one",
    "image_split": "one_to_one",
    "class_labels": "many_to_one",
}
