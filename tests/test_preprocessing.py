import pytest
import pylidc as pl
from deepl_lidc.preprocessing import extract_nodules

def test_extract_nodules():
    test_patient_id = 'LIDC-IDRI-0003'
    
    scans = pl.query(pl.Scan).filter(pl.Scan.patient_id == test_patient_id).first()
    nods = scans.cluster_annotations()
    n_nods = len(nods)

    # Call the function with the test patient_id
    result = extract_nodules(test_patient_id)
    
    # Assert the expected output type. 
    assert isinstance(result, dict)
    assert len(result) == n_nods
    print(result)