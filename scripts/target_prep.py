"""
Script to prepare the target data for training and testing.
"""
from deepl_lidc.preprocessing import available_patients, prepare_patient_label, prepare_nodule_label

# Path to the data directory
PATH_scans = "/home/vadim/Development/Projects/DeepL-LIDC/data/LIDC-IDRI/"

# Path to the target data directory
PATH_target = "/home/vadim/Development/Projects/DeepL-LIDC/data/final-lidc-nodule-semantic-scores.csv"


if __name__ == "__main__":

    ids = available_patients(PATH_scans)
    prepare_patient_label(PATH_target, ids)
    prepare_nodule_label(PATH_target)
    