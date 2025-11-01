import plotly.express as px
import pandas as pd
from typing import List, Dict, Tuple

# Perform split - calls merge_data() and then creates the splits
def create_test_split(dataset_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create official train-test split for the CUB-200-2011 as dfs with image_path information

    Args:
        dataset_dir (Path): Path to the folder containing the metadata and train-test split.
    
    Returns:
        (train_df, test_df): Two DataFrames with columns:
            ['image_id', 'image_path', 'class_id', 'class_name', 'is_training']
    """
    # Create merged df
    merged_df = merge_metadata(dataset_dir)

    # Create train-test split
    train_df = merged_df[merged_df["is_training"] == 1].reset_index(drop=True)
    test_df = merged_df[merged_df["is_training"] == 0].reset_index(drop=True)

    print(f"Total images: {len(merged_df)}")
    print(f"Train: {len(train_df)} | Test: {len(test_df)} | Train_Classes: {train_df['class_id'].nunique()}| Test_Classes: {test_df['class_id'].nunique()}")
    return train_df, test_df


# Create validation split - input train images only
def create_validation_split(
    train_df: pd.DataFrame,
    val_fraction: float = 0.2,
    random_state: int = 42,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the official training set into training and validation subsets.

    Args:
        train_df (pd.DataFrame): training DataFrame (from create_test_split()).
        val_fraction (float): Fraction of training data to use for validation (default=0.2).
        random_state (int): Random seed for reproducibility.
        stratify (bool): Preserve class distribution if True (default=True).

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (train_sub_df, val_sub_df)
    """
    stratify_labels = train_df["class_id"] if stratify else None

    train_sub_df, val_sub_df = train_test_split(
        train_df,
        test_size=val_fraction,
        random_state=random_state,
        stratify=stratify_labels
    )

    train_sub_df = train_sub_df.reset_index(drop=True)
    val_sub_df = val_sub_df.reset_index(drop=True)

    print(f"🔹 Training subset: {len(train_sub_df)} images, "
          f"{train_sub_df['class_id'].nunique()} classes")
    print(f"🔹 Validation subset: {len(val_sub_df)} images, "
          f"{val_sub_df['class_id'].nunique()} classes")

    return train_sub_df, val_sub_df


# Define function to check that there are no overlapping images between train, validation, and test sets
def check_no_overlap(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """
    Checks that there are no overlapping images between train, validation, and test sets using set operator &
    """
    train_ids = set(train_df['image_id'])
    val_ids = set(val_df['image_id'])
    test_ids = set(test_df['image_id'])

    # Compute intersections between sets with &
    overlap_train_val = train_ids & val_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = val_ids & test_ids

    assert not overlap_train_val, f"Overlap between train and val: {overlap_train_val}"
    assert not overlap_train_test, f"Overlap between train and test: {overlap_train_test}"
    assert not overlap_val_test, f"Overlap between val and test: {overlap_val_test}"

    print("No overlapping image_ids between train, validation, and test sets.")


# Define function to plot class balance
def plot_class_balance(dfs: dict[str, pd.DataFrame]) -> None:
    """
    Plot cumulative class distribution for each dataset split using Plotly.

    Args:
        dfs (dict[str, pd.DataFrame]): Dictionary of DataFrames, e.g.
            {"train": train_df, "val": val_df, "test": test_df}
    """
    plot_data = []

    for name, df in dfs.items():
        # Count images per class
        class_counts = (
            df["class_id"]
            .value_counts()
            .sort_values()
            .reset_index()
        )

        # Compute cumulative percentage
        class_counts["cum_percent"] = (class_counts["count"].cumsum() / class_counts["count"].sum()) * 100
        class_counts["rank"] = range(1, len(class_counts) + 1)
        class_counts["dataset"] = name

        plot_data.append(class_counts)

    combined_df = pd.concat(plot_data, ignore_index=True)

    # Plot cumulative distribution
    fig = px.line(
        combined_df,
        x="rank",
        y="cum_percent",
        color="dataset",
        markers=True,
        title="Cumulative Class Distribution Across Splits",
        labels={
            "rank": "Class Rank (sorted by frequency)",
            "cum_percent": "Cumulative % of Images",
            "dataset": "Dataset Split"
        }
    )

    fig.update_layout(
        yaxis=dict(range=[0, 100]),
        xaxis=dict(title="Class Rank (1–200)"),
        template="plotly_white",
        legend=dict(title="Split", orientation="h", y=-0.25, x=0.3)
    )

    fig.show()
