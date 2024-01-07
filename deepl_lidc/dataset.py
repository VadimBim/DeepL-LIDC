""" 
Module for dataset and dataloader classes
"""
import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import pandas as pd


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



def repair_dict_names(dict):
    """mapping key values of dictionary to delate values that inclue "_0" 
    
    """
    new_dict = {}
    for old_key, val in dict.items():
        new_key = old_key.replace('_0', '')
        new_dict[new_key] = val
    return new_dict

class NoduleDataset(Dataset):
    def __init__(self, target_file, nodule_dir, transform=normalize, 
                 target_transform=F.one_hot):

        self.nodule_dir = nodule_dir
        self.transform = transform
        self.target_transform = target_transform

        ##read csv file as dictionary: {'LIDC-IDRI-0902-2': 1, ..}
        df = pd.read_csv(target_file)
        image_targets = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        ##read names of all nodules without extentions
        available_nodules=[nodule.split('.')[0] for nodule in sorted(os.listdir(nodule_dir))]

        image_targets=repair_dict_names(image_targets)

        #create an array with tuples (image,label) where image and label are tensors
        self.res = []
        target_size = (84, 84) ##size of tensor
        for x in available_nodules:
            nodule_path = os.path.join(self.nodule_dir, x + '.npy')
            nodule = torch.from_numpy(np.load(nodule_path))
            #extract middle slice of the nodule
            image = nodule[:, :, nodule.shape[2]//2]
            #check if the image is empty (all values are equal to each other)
            if torch.allclose(image, image[0, 0]):
                continue
            paded_image = pad_image(image, image.min(), target_size)

            label = image_targets[x] - 1 # 1-5 -> 0-4, necessary for one_hot encoding
            label = torch.tensor(label)

            if self.transform:
                paded_image = self.transform(paded_image)
            if self.target_transform:
                label = self.target_transform(label, num_classes=5)
                label = label.type(torch.float32)

            self.res.append((paded_image, label))

    def __len__(self):
        return len(self.res)
    
    def __getitem__(self, index) -> tuple:
        return self.res[index]
    
    @property
    def labels(self):
        return [label for _, label in self.res]
    
    def get_label_by_index(self,index):
        return torch.argmax(self.res[index][1]).item()
    
    def get_index_name(self,index):
        return self.available_nodules[index]
    
    def add_synthetic_by_rotations(self,target_count,rotations=[1,2,3]):
        first_rotated_indexes=[] #array with indexes of first data that was rotated. (used to debuging)
        label_counts = [0,0,0,0,0]
        for data in self.res:
            label = torch.argmax(data[1]).item()
            label_counts[label]+=1
        print(f"Label counts in dataset befor adding synthetic data by rotations: {label_counts}")

        for i in range(len(self.res)):
            label = self.get_label_by_index(i)
            if label_counts[label] < target_count:
                self.res.append(self.res[i])
                if first_rotated_indexes==[]:
                    first_rotated_indexes=[i,len(self.res),len(self.res)+1, len(self.res)+2]
                for deg in rotations:
                    rotated= np.rot90(self.res[i][0],k=deg)
                    self.res.append((torch.tensor(rotated.copy()),self.res[i][1]))
                label_counts[label]+=len(rotations)
        print(f"Label counts in dataset after adding synthetic data by rotations: {label_counts}")

        return first_rotated_indexes
    
    def add_synthetic_by_flips(self, target_count, flips=['horizontal', 'vertical', 'center']):
        first_flipped_indexes = []  #array with indexes of first data that was rotated. (used to debuging)  ``
        label_counts = [0, 0, 0, 0, 0]
        
        for data in self.res:
            label = torch.argmax(data[1])
            label_counts[label] += 1
        
        print(f"Label counts in dataset before adding synthetic data by flips: {label_counts}")

        for i in range(len(self.res)):
            label = self.get_label_by_index(i)
            if label_counts[label] < target_count:
                self.res.append(self.res[i])
                if not first_flipped_indexes:
                    first_flipped_indexes = [i, len(self.res), len(self.res) + 1, len(self.res) + 2]
                
                for flip_type in flips:
                    if flip_type == 'horizontal':
                        flipped = np.fliplr(self.res[i][0])
                    elif flip_type == 'vertical':
                        flipped = np.flipud(self.res[i][0])
                    elif flip_type == 'center':
                        flipped = np.flipud(np.fliplr(self.res[i][0]))
                 #   elif flip_type == 'diagonal':
                #        flipped = np.transpose(self.res[i][0])
                    else:
                        raise ValueError(f"Unsupported flip type: {flip_type}")
                    
                    self.res.append((torch.tensor(flipped.copy()), self.res[i][1]))
                
                label_counts[label] += len(flips)

        print(f"Label counts in dataset after adding synthetic data by flips: {label_counts}")
        
        return first_flipped_indexes