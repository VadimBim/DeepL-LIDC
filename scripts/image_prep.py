""" 
Script to extract all the nodules from dataset
and save them as .npy files in the data folder
"""
import os
import numpy as np
from deepl_lidc.preprocessing import extract_nodules, available_patients

path_scans = '/home/vadim/Development/Projects/DeepL-LIDC/data/LIDC-IDRI'
path_nodules = '/home/vadim/Development/Projects/DeepL-LIDC/data/nodules'

def main():
    patient_ids = available_patients(path_scans)
    #for all available patient_ids extract the nodules and save them as .npy files
    #with the name 'LIDC-IDRI-XXXX-nodule_id.npy'
    for patient_id in patient_ids:
        nodules = extract_nodules(patient_id)
        for nodule_id, nodule in nodules.items():
            np.save(os.path.join(path_nodules, nodule_id), nodule)

if __name__ == '__main__':
    main()
