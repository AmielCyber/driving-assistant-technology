"""
CHAT GPT GENERATED CODE
Prompt: 
    I have a csv table with this columns

    frame-number,predicted-left-lane,predicted-right-lane,predicted-left-status,predicted-right-status,actual-left-lane,actual-right-lane,actual-left-stat,actual-right-stat

    with left-lane and right-lane containing a boolean value 
    and left-status and right-status containing safe, warning, and alert values

    how can I create a python script to display a confusion matrix and calculate precision and recall. Using libraries can be great
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


STATUS_LABELS = ["safe", "warning", "alert"]


def normalize_boolean_column(series: pd.Series) -> pd.Series:
    """
    Converts common CSV Boolean representations into actual bool values.

    Accepted true values:
        true, 1, yes, y

    Accepted false values:
        false, 0, no, n
    """
    mapping = {
        "true": True,
        "1": True,
        "yes": True,
        "y": True,
        "false": False,
        "0": False,
        "no": False,
        "n": False,
    }

    normalized = series.astype(str).str.strip().str.lower()
    result = normalized.map(mapping)

    if result.isna().any():
        invalid_values = sorted(normalized[result.isna()].unique())
        raise ValueError(
            f"Invalid Boolean values in column '{series.name}': "
            f"{invalid_values}"
        )

    return result.astype(bool)


def normalize_status_column(series: pd.Series) -> pd.Series:
    """Normalizes and validates safe/warning/alert status values."""
    normalized = series.astype(str).str.strip().str.lower()

    invalid_mask = ~normalized.isin(STATUS_LABELS)

    if invalid_mask.any():
        invalid_values = sorted(normalized[invalid_mask].unique())
        raise ValueError(
            f"Invalid status values in column '{series.name}': "
            f"{invalid_values}"
        )

    return normalized


def display_confusion_matrix(
    actual: pd.Series,
    predicted: pd.Series,
    labels: list,
    title: str,
    output_path: Path,
) -> None:
    matrix = confusion_matrix(actual, predicted, labels=labels)

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )

    plt.title(title)
    plt.xlabel("predicted value")
    plt.ylabel("Actual value")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()
    plt.close()


def evaluate_binary_lane(
    dataframe: pd.DataFrame,
    actual_column: str,
    predicted_column: str,
    lane_name: str,
    output_directory: Path,
) -> None:
    actual = dataframe[actual_column]
    predicted = dataframe[predicted_column]

    precision, recall, f1, _ = precision_recall_fscore_support(
        actual,
        predicted,
        average="binary",
        pos_label=True,
        zero_division=0,
    )

    accuracy = accuracy_score(actual, predicted)

    print(f"\n{'=' * 60}")
    print(f"{lane_name} lane detection")
    print(f"{'=' * 60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            actual,
            predicted,
            labels=[False, True],
            target_names=["lane absent", "lane present"],
            zero_division=0,
        )
    )

    display_confusion_matrix(
        actual=actual,
        predicted=predicted,
        labels=[False, True],
        title=f"{lane_name} Lane Detection",
        output_path=output_directory
        / f"{lane_name.lower()}_lane_confusion_matrix.png",
    )


def evaluate_lane_status(
    dataframe: pd.DataFrame,
    actual_column: str,
    predicted_column: str,
    lane_name: str,
    output_directory: Path,
) -> None:
    actual = dataframe[actual_column]
    predicted = dataframe[predicted_column]

    print(f"\n{'=' * 60}")
    print(f"{lane_name} lane status")
    print(f"{'=' * 60}")

    print(
        classification_report(
            actual,
            predicted,
            labels=STATUS_LABELS,
            zero_division=0,
            digits=4,
        )
    )

    macro_precision, macro_recall, macro_f1, _ = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=STATUS_LABELS,
            average="macro",
            zero_division=0,
        )
    )

    weighted_precision, weighted_recall, weighted_f1, _ = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=STATUS_LABELS,
            average="weighted",
            zero_division=0,
        )
    )

    print("Multiclass averages:")
    print(f"Macro precision:    {macro_precision:.4f}")
    print(f"Macro recall:       {macro_recall:.4f}")
    print(f"Macro F1:           {macro_f1:.4f}")
    print(f"Weighted precision: {weighted_precision:.4f}")
    print(f"Weighted recall:    {weighted_recall:.4f}")
    print(f"Weighted F1:        {weighted_f1:.4f}")

    display_confusion_matrix(
        actual=actual,
        predicted=predicted,
        labels=STATUS_LABELS,
        title=f"{lane_name} Lane Status",
        output_path=output_directory
        / f"{lane_name.lower()}_status_confusion_matrix.png",
    )


def main() -> None:
    csv_path = Path("lane-departure-log.csv")
    output_directory = Path("evaluation-results")
    output_directory.mkdir(parents=True, exist_ok=True)

    dataframe = pd.read_csv(csv_path)

    required_columns = [
        "frame-number",
        "predicted-left-lane",
        "predicted-right-lane",
        "predicted-left-status",
        "predicted-right-status",
        "actual-left-lane",
        "actual-right-lane",
        "actual-left-stat",
        "actual-right-stat",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"CSV is missing required columns: {missing_columns}"
        )

    # Remove rows that are incomplete for any evaluated value.
    dataframe = dataframe.dropna(subset=required_columns[1:]).copy()

    boolean_columns = [
        "predicted-left-lane",
        "predicted-right-lane",
        "actual-left-lane",
        "actual-right-lane",
    ]

    status_columns = [
        "predicted-left-status",
        "predicted-right-status",
        "actual-left-stat",
        "actual-right-stat",
    ]

    for column in boolean_columns:
        dataframe[column] = normalize_boolean_column(dataframe[column])

    for column in status_columns:
        dataframe[column] = normalize_status_column(dataframe[column])

    evaluate_binary_lane(
        dataframe=dataframe,
        actual_column="actual-left-lane",
        predicted_column="predicted-left-lane",
        lane_name="Left",
        output_directory=output_directory,
    )

    evaluate_binary_lane(
        dataframe=dataframe,
        actual_column="actual-right-lane",
        predicted_column="predicted-right-lane",
        lane_name="Right",
        output_directory=output_directory,
    )

    evaluate_lane_status(
        dataframe=dataframe,
        actual_column="actual-left-stat",
        predicted_column="predicted-left-status",
        lane_name="Left",
        output_directory=output_directory,
    )

    evaluate_lane_status(
        dataframe=dataframe,
        actual_column="actual-right-stat",
        predicted_column="predicted-right-status",
        lane_name="Right",
        output_directory=output_directory,
    )


if __name__ == "__main__":
    main()
