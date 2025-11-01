# Intrinsic constants for CUB_200_2011 if downloaded from "https://data.caltech.edu/records/65de6-vp158/files/CUB_200_2011.tgz?download=1"
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
