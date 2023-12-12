""" 
Module with functions that give information
about specific patient. print statements or
plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import pylidc as pl

def nodule_info(pid):
    """Print information about nodules in patient.

    Args:
        pid (string): Patient id of the form "LIDC-IDRI-XXXX".
    """
    
    scan = pl.query(pl.Scan).filter(pl.Scan.patient_id == pid).first()
    
    nods = scan.cluster_annotations()
    #print number of nodules
    print("%s has %d nodules." % (scan, len(nods)))

    #print number of annotations for each nodule
    for i,nod in enumerate(nods):
        print("Nodule %d has %d annotations." % (i+1, len(nods[i])))

def plot_slice(vol, slice_num):
    """
    Plot a slice of the volume
    
    Args:
        vol (numpy array): Volume of the patient.
        slice_num (int): Slice number to plot.
    """    
    plt.figure(figsize=(5, 5))
    plt.imshow(vol[:,:,slice_num])
    plt.show()
    
def show_nodule(vol, bbox, mask, z_slice):
    """Show the nodule in the volume

    Args:
        vol (np.array): 3D volume (512 x 512 x n_slices)
        bbox (_type_) : slice of the volume used to detect where the nodule is
        mask (_type_): mask of the nodule
        z_slice (int, optional): vertical slice of the volume to show.
    """
    
    vmin = vol.min()
    vmax = vol.max()
    
    minivol = vol[bbox]
    #keep only the slice where the nodule is
    nodule_vol = np.where(mask == 1, minivol, vmin)

    fig,ax = plt.subplots(1,3,figsize=(5,3))

    ax[0].imshow(minivol[:,:,z_slice], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)

    ax[1].imshow(nodule_vol[:,:,z_slice], cmap=plt.cm.gray, vmin=vmin, vmax=vmax)

    ax[2].imshow(mask[:,:,z_slice], cmap=plt.cm.gray)

    #hide the axis
    for a in ax:
        a.axis('off')

    plt.tight_layout()
    #set title
    plt.show()