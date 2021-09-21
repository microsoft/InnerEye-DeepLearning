#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import json
from pathlib import Path

import pandas as pd
import numpy as np


def create_nih_dataframe(mapping_file_path: Path) -> pd.DataFrame:
    """
    This function loads the json file mapping NIH ids to Kaggle images.
    Loads the original NIH label (multiple labels for each image).
    Then it creates the grouping by NIH categories (pneumonia, pneumonia like,
    other disease, no finding).
    :param mapping_file_path: path to the json mapping from NIH to Kaggle dataset (on the
    RSNA webpage)
    :return: dataframe with original NIH labels for each patient in the Kaggle dataset.
    """
    with open(mapping_file_path) as f:
        list_subjects = json.load(f)
    orig_dataset = pd.DataFrame(columns=["subject", "orig_label"])
    orig_dataset["subject"] = [l["subset_img_id"] for l in list_subjects]  # noqa: E741
    orig_labels = [str(l['orig_labels']).lower() for l in list_subjects]  # noqa: E741
    orig_dataset["nih_pneumonia"] = ["pneumonia" in l for l in orig_labels]  # noqa: E741
    orig_dataset["nih_pneumonia_like"] = [(("infiltration" in l or "consolidation" in l) and ("pneumonia" not in l)) for
                                          l in orig_labels]  # noqa: E741
    orig_dataset["no_finding"] = ["no finding" in str(l).lower() for l in orig_labels]  # noqa: E741
    orig_dataset["orig_label"] = orig_labels
    orig_dataset["orig_label"].apply(lambda x: sorted(x))
    orig_dataset[
        "nih_other_disease"] = ~orig_dataset.nih_pneumonia_like & ~orig_dataset.nih_pneumonia & ~orig_dataset.no_finding
    orig_dataset[
        "nih_category"] = 1 * orig_dataset.nih_pneumonia + 2 * orig_dataset.no_finding + 3 * \
                          orig_dataset.nih_other_disease
    orig_dataset["StudyInstanceUID"] = [l["StudyInstanceUID"] for l in list_subjects]  # noqa: E741
    orig_dataset.nih_category = orig_dataset.nih_category.apply(lambda x: ["Consolidation/Infiltration", "Pneumonia",
                                                                           "No finding", "Other disease"][x])
    return orig_dataset


def process_detailed_probs_dataset(detailed_probs_path: Path) -> pd.DataFrame:
    """
    This function loads the csv file with the detailed information for each bounding boxes as annotated
    by the readers during the adjudication for the preparation of the challenge. It maps low, medium and high
    probabilities label to a numerical scale from 1 to 3. Computes the minimum, maximum, average confidence for each
    patient for which at least one bounding box was present.
    :param detailed_probs_path: path to detailed_probs csv file released in Kaggle challenge.
    :return: dataframe with metrics for confidence in bounding boxes by patient.
    """
    conversion_map = {"Lung Opacity (Low Prob)": 1, "Lung Opacity (Med Prob)": 2, "Lung Opacity (High Prob)": 3}
    detailed_probs_dataset = pd.read_csv(detailed_probs_path)
    detailed_probs_dataset["ClassProb"] = detailed_probs_dataset["labelName"].apply(lambda x: conversion_map[x])
    process_details = detailed_probs_dataset.groupby("StudyInstanceUID")["ClassProb"].agg(
        [np.mean, np.min, np.max, np.count_nonzero, list])
    process_details.rename(columns={"mean": "avg_conf_score", "amin": "min_conf_score", "amax": "max_conf_score"},
                           inplace=True)
    return process_details


def create_mapping_dataset_nih(mapping_file_path: Path,
                               kaggle_dataset_path: Path,
                               detailed_class_info_path: Path,
                               detailed_probs_path: Path) -> pd.DataFrame:
    """
    Creates the final chest x-ray dataset combining labels from NIH, kaggle and detailed information about kaggle
    labels from the detailed_class_info and detailed_probs csv file released during the challenge.
    :param mapping_file_path:
    :param kaggle_dataset_path:
    :param detailed_class_info_path:
    :param detailed_probs_path:
    :return: detailed dataset
    """
    orig_dataset = create_nih_dataframe(mapping_file_path)
    kaggle_dataset = pd.read_csv(kaggle_dataset_path)
    difficulty = pd.read_csv(detailed_class_info_path).drop_duplicates()
    detailed_probs_dataset = process_detailed_probs_dataset(detailed_probs_path)
    # Merge NIH info with Kaggle dataset
    merged = pd.merge(orig_dataset, kaggle_dataset)
    merged.rename(columns={"label": "label_kaggle"}, inplace=True)
    # Define binary label from original NIH label, for consolidation/infiltration
    # mapping is not clear, use kaggle label.
    merged.loc[merged.nih_pneumonia, "binary_nih_initial_label"] = True
    merged.loc[merged.no_finding | merged.nih_other_disease, "binary_nih_initial_label"] = False
    merged.loc[merged.nih_pneumonia_like, "binary_nih_initial_label"] = merged.loc[
        merged.nih_pneumonia_like, "label_kaggle"]
    # Add subclass information from Kaggle challenge to define ambiguous cases
    merged = pd.merge(merged, difficulty, left_on="subject", right_on="patientId")
    merged["not_normal"] = (merged["class"] == "No Lung Opacity / Not Normal")
    # Add difficulty information from Kaggle based on presence of bounding boxes
    merged = pd.merge(merged, detailed_probs_dataset, on="StudyInstanceUID", how="left")
    merged.drop(columns=["patientId"], inplace=True)
    merged.fillna(-1, inplace=True)
    # Ambiguous if there was only low probability boxes and the adjudicated label is true
    merged.loc[merged.label_kaggle, ["ambiguous"]] = (merged.loc[merged.label_kaggle].min_conf_score == 1) & (
            merged.loc[merged.label_kaggle].max_conf_score == 1)
    # Ambiguous if there was some bounding boxes but adjudicated label is false
    merged.loc[~merged.label_kaggle, ["ambiguous"]] = merged.loc[~merged.label_kaggle].min_conf_score > -1
    return merged


if __name__ == "__main__":
    from default_paths import INNEREYE_DQ_DIR
    current_dir = Path(__file__).parent
    mapping_file = Path(__file__).parent / "pneumonia-challenge-dataset-mappings_2018.json"
    kaggle_dataset_path = current_dir / "stage_2_train_labels.csv"
    detailed_class_info = current_dir / "stage_2_detailed_class_info.csv"
    detailed_probs = current_dir / "RSNA_pneumonia_all_probs.csv"
    dataset = create_mapping_dataset_nih(mapping_file, kaggle_dataset_path, detailed_class_info, detailed_probs)
    dataset.to_csv(INNEREYE_DQ_DIR / "datasets" / "noisy_chestxray_dataset.csv", index=False)
