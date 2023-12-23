""" 
Script to extract volume of annotations and save to csv file
"""
import pylidc as pl
from deepl_lidc.preprocessing import available_patients

path_scans = '/home/vadim/Development/Projects/DeepL-LIDC/data/LIDC-IDRI'

path_save = '/home/vadim/Development/Projects/DeepL-LIDC/data/ann_volumes.csv'

patiant_ids = available_patients(path_scans)

def extract_ann_volumes(path_save):
    """Function to extract volume of annotations and save to csv file

    Args:
        path_save (string): path to save csv file
    """
    anns = pl.query(pl.Annotation).all()
    ann_volumes = [ann.volume for ann in anns]
    ann_ids = [ann.id for ann in anns]
    #save to csv file
    with open(path_save, 'w') as f:
        f.write('annotation_id, volume\n')
        for ann_id, ann_volume in zip(ann_ids, ann_volumes):
            f.write('{}, {}\n'.format(ann_id, ann_volume))

def main():
    extract_ann_volumes(path_save)

if __name__ == '__main__':
    main()