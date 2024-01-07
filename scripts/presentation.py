"""Script to visualize slices for presentation"""
import pylidc as pl 

def main():
    
    pid = "LIDC-IDRI-0013"

    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    nods = scan.cluster_annotations()
    scan.visualize(annotation_groups=nods)


if __name__ == "__main__":
    main()