""" 
Module for dataset and dataloader classes
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

def pad_image(input_tensor, padding_value, target_size):
    """Pad tensor to target size keeping the initial tensor in the middle.
    Args:
        input_tensor (torch.tensor): tensor to pad
        padding_value (float): value to pad with
        target_size (tuple): target size of the returned tensor

    Returns:
        torch.tensor: padded tensor
    """
    current_size = list(input_tensor.shape)
    padding_needed = [target_size[i] - current_size[i] for i in range(len(current_size))]
    
    if all(padding <= 0 for padding in padding_needed):
        return input_tensor
    
    padding_before = [padding // 2 for padding in padding_needed]
    padding_after = [padding - padding_before[i] for i, padding in enumerate(padding_needed)]
    
    padded_tensor = F.pad(input_tensor, (padding_before[1], padding_after[1], padding_before[0], padding_after[0]), value=padding_value)
    return padded_tensor

def normalize(tensor):
    """normalize tensor to [0, 1]

    Args:
        tensor (torch.tensor): input tensor

    Returns:
        torch.tensor: normalized tensor
    """
    return (tensor - tensor.min()) / (tensor.max() - tensor.min())

class NoduleDataset(Dataset):
    def __init__(self, target_file, nodule_dir, transform=normalize, 
                 target_transform=F.one_hot):
        self.image_targets = pd.read_csv(target_file)
        self.nodule_dir = nodule_dir
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.image_targets)
    
    def __getitem__(self, index) -> tuple:
        nodule_path = os.path.join(self.nodule_dir, 
                                  self.image_targets.iloc[index, 0] + '.npy')
        nodule = torch.from_numpy(np.load(nodule_path))
        #extract middle slice of the nodule
        image = nodule[:, :, nodule.shape[2]//2]
        target_size = (64, 64)
        paded_image = pad_image(image, image.min(), target_size)
        label = self.image_targets.iloc[index, 1] - 1 # 1-5 -> 0-4, necessary for one_hot encoding
        label = torch.tensor(label)
        
        if self.transform:
            paded_image = self.transform(paded_image)
        if self.target_transform:
            label = self.target_transform(label, num_classes=5)
            
        return paded_image, label
        
            
        

        
        

