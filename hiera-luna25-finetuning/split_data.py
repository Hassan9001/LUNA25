# Script for splitting data into training and validation sets based on PatientID,
# maintaining label ratios using StratifiedKFold

import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedKFold


def create_patient_level_splits(input_csv_path, output_dir="data_splits", n_splits=5, random_state=2025):
    """
    Split data based on PatientID, maintain label ratios using StratifiedKFold,
    and sort output CSVs by AnnotationID.

    Args:
        input_csv_path (str or Path): Path to the original CSV file.
        output_dir (str or Path): Directory to save the training and validation CSV files.
        n_splits (int): Number of folds for StratifiedKFold.
        random_state (int): Random seed for StratifiedKFold.
    """
    input_csv_path = Path(input_csv_path)
    output_dir = Path(output_dir)

    print(f"Loading data from: {input_csv_path}")
    try:
        df = pd.read_csv(input_csv_path)
    except FileNotFoundError:
        print(
            f"Error: File not found at {input_csv_path}. Please check the path.")
        return
    except Exception as e:
        print(f"An error occurred while loading the CSV file: {e}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory set to: {output_dir}")

    # Extract unique PatientIDs and their corresponding patient-level labels.
    # If any nodule for a patient has label 1, the patient's overall label is 1.
    # Otherwise, if all nodules for a patient have label 0, the patient's overall label is 0.
    patient_labels = df.groupby('PatientID')['label'].max().reset_index()
    unique_patient_ids = patient_labels['PatientID'].values
    patient_level_labels = patient_labels['label'].values

    print(f"#Patients: {len(unique_patient_ids)}")
    print("Patient-level label distribution:")
    print(pd.Series(patient_level_labels).value_counts(normalize=True))
    print(f"#Nodules: {len(df['AnnotationID'].unique())}")
    print("Nodule-level label distribution:")
    print(df['label'].value_counts(normalize=True))

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                          random_state=random_state)

    print(
        f"Starting to generate {n_splits} pairs of train/validation splits...")
    for fold, (train_index, val_index) in enumerate(skf.split(unique_patient_ids, patient_level_labels)):
        print(f"\n--- Processing Fold {fold + 1}/{n_splits} ---")

        train_patient_ids = unique_patient_ids[train_index]
        val_patient_ids = unique_patient_ids[val_index]

        train_df = df[df['PatientID'].isin(train_patient_ids)].copy()
        val_df = df[df['PatientID'].isin(val_patient_ids)].copy()

        common_patient_ids = set(train_df['PatientID']).intersection(
            set(val_df['PatientID']))
        if common_patient_ids:
            print(
                f"Warning: Data leakage detected in Fold {fold + 1}! Common PatientIDs: {common_patient_ids}")
        else:
            print(
                f"Fold {fold + 1}: Verified, no PatientID overlap between train and validation sets.")

        print(f"Fold {fold + 1} Training set #nodules: {len(train_df)}")
        print(f"Fold {fold + 1} Training set nodule-level label distribution:")
        print(train_df['label'].value_counts(normalize=True))

        print(f"Fold {fold + 1} Validation set #nodules: {len(val_df)}")
        print(f"Fold {fold + 1} Validation set nodule-level label distribution:")
        print(val_df['label'].value_counts(normalize=True))

        if 'AnnotationID' in train_df.columns:
            train_df.sort_values(
                by='AnnotationID', ascending=True, inplace=True)
            print("Training set sorted by AnnotationID.")
        else:
            print(
                "Warning: 'AnnotationID' column not found in training set. Skipping sort.")

        if 'AnnotationID' in val_df.columns:
            val_df.sort_values(by='AnnotationID', ascending=True, inplace=True)
            print("Validation set sorted by AnnotationID.")
        else:
            print(
                "Warning: 'AnnotationID' column not found in validation set. Skipping sort.")

        train_output_path = output_dir / f"train_fold_{fold + 1}.csv"
        val_output_path = output_dir / f"val_fold_{fold + 1}.csv"

        train_df.to_csv(train_output_path, index=False)
        val_df.to_csv(val_output_path, index=False)

        print(f"Saved training set to: {train_output_path}")
        print(f"Saved validation set to: {val_output_path}")

    print("\nAll folds' data splitting completed!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Split LUNA25 annotation data at the patient level.")
    parser.add_argument(
        "--annotation_csv_path",
        type=str,
        default="LUNA25_Public_Training_Development_Data.csv",
        help="Path to the input annotation CSV file (default: %(default)s)"
    )
    args = parser.parse_args()
    create_patient_level_splits(
        input_csv_path=args.annotation_csv_path,
        n_splits=5,
        random_state=2025
    )
