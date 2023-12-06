""" 
A module with functions for preprocessing the target data.
"""

import os
import numpy as np
import pandas as pd

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