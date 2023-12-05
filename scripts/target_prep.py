"""
Script to prepare the target data for training and testing.
"""

import os
import pandas as pd

# Path to the data directory
PATH_scans = "/home/vadim/Development/Projects/DeepL-LIDC/data/LIDC-IDRI/"

#save patiants ids from PATH_scans to a list
#ids are of the form LIDC-IDRI-XXXX
#ignore LICENSE
ids = [id for id in os.listdir(PATH_scans) if id != 'LICENSE']


# Path to the target data directory
PATH = "/home/vadim/Development/Projects/DeepL-LIDC/data/final-lidc-nodule-semantic-scores.csv"
df = pd.read_csv(PATH)

if __name__ == "__main__":
    df = df.drop(columns=['subtlety', 'internalStructure', 
                        'calcification', 'sphericity', 'margin', 'lobulation', 'spiculation', 'texture'])

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
    
    #keep only the patients that are in the ids list
    df = df[df['patient_id'].isin(ids)]
    
    #save the dataframe to a csv file'
    df.to_csv('/home/vadim/Development/Projects/DeepL-LIDC/data/target.csv', index=False)
