""" 
A module with functions for preprocessing the data.
"""

import os
import pandas as pd

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
    
    #add a new column to the dataframe named 'target'.
    #for each patient sum the malignancy scores and
    # divide by the number of nodules.
    # if the score is >= 3 then the patient is malignant
    df['target'] = df.groupby('patient_id')['malignancy'].transform('sum') / df.groupby('patient_id')['malignancy'].transform('count')
    df['target'] = df['target'].apply(lambda x: 1 if x >= 3 else 0)
    df = df.drop(columns=['malignancy'])
    #drop anotation_id and nodule column
    df = df.drop(columns=['annotation_id', 'nodule'])
    #drop patiants duplicates
    df = df.drop_duplicates(subset=['patient_id'])
    
    
    #save the dataframe to a csv file where the raw csv file is located
    save_path = os.path.dirname(path_raw_label)
    df.to_csv(os.path.join(save_path, 'target.csv'), index=False)