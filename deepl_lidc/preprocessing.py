""" 
A module with functions for preprocessing raw data.
"""

import os
import numpy as np
import pandas as pd
import pylidc as pl
from pylidc.utils import consensus

#target preprocessing
def sigmoid(x, alpha=1.0):
    return 1 / (1 + np.exp(-alpha*x))

def available_patients(path_scans):
    """
    Returns a list of the available patients.
    """
    ids = [id for id in os.listdir(path_scans) if id != 'LICENSE']
    return ids

def prepare_label(path_raw_label, ids):
    """function for preparing the label data.

    Args:
        path_raw_label (string): path to the raw label csv file
        ids (list): list of the available patients
    """
    
    df = pd.read_csv(path_raw_label)

    df = df.drop(columns=['subtlety', 'internalStructure', 
                        'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture'])

    #keep only the patients that are in the ids list
    df = df[df['patient_id'].isin(ids)]
    
    # Calculate the mean malignancy score for each nodule
    df['mean_malignancy'] = df.groupby(['patient_id', 'nodule'])['malignancy'].transform('mean')

    # Count the number of nodules for each patient
    df['num_nodules'] = df.groupby('patient_id')['nodule'].transform('nunique')

    # Apply the formula to compute 'target' for each row
    df['target'] = sigmoid(df['num_nodules'] * (df['mean_malignancy'] / 1.5 - 1), alpha=0.6)

    #drop the columns that are not needed anymore
    df = df.drop(columns=['nodule', 'annotation_id', 'malignancy', 'mean_malignancy', 'num_nodules'])

    #drop patiants duplicates
    df = df.drop_duplicates(subset=['patient_id'])
    
    
    #save the dataframe to a csv file where the raw csv file is located
    save_path = os.path.dirname(path_raw_label)
    df.to_csv(os.path.join(save_path, 'target.csv'), index=False)
    
#Xs preprocessing

def extract_nodules(patient_id):
    """Extracts the nodules from a given patient.

    Args:
        patient_id (str): id of the patient of the form 'LIDC-IDRI-XXXX'
    
    returns:
        nodules (dictionary): dictionary of the form {'LIDC-IDRI-XXXX-nodule_id': nodule}
        nodule is a 3d numpy array
        nodule_id is just a number from 1 to the number of nodules in the patient
    """
    
    # Query for a scan, and convert it to an array volume (512 x 512 x n_slices).
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == patient_id).first()
    vol = scan.to_volume()
    #minimum value of the volume which corresponds to the background
    vmin = -2048

    # Cluster the annotations for the scan
    nods = scan.cluster_annotations()
    num_nods = len(nods)

    #initialize the dictionary of nodules with filled keys
    nodules = {patient_id + '-' + str(i): None for i in range(1, num_nods + 1)}
    # Perform a consensus consolidation and 50% agreement level.
    for i, anns in enumerate(nods, start=1):
        cmask, cbbox = consensus(anns, clevel=0.5, ret_masks=False)
        minivol = vol[cbbox]
        filtered_vol = np.where(cmask == 1, minivol, vmin)
        nodules[patient_id + '-' + str(i)] = filtered_vol

    return nodules
    