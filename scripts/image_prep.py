""" 
A script to extract the middle
slice of a 3D image and save it
"""

import os
import pylidc as pl

pid = 'LIDC-IDRI-0012'
scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()

nods = scan.cluster_annotations()

print("%s has %d nodules." % (scan, len(nods)))

for i,nod in enumerate(nods):
    print("Nodule %d has %d annotations." % (i+1, len(nods[i])))
    
vol = scan.to_volume()
print(vol.shape)

scan.visualize(annotation_groups=nods)