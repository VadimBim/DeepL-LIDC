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
import random
from imblearn.over_sampling import RandomOverSampler
import copy



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

        self.max_x = 0
        self.max_y = 0
        self.index_max_x = -1
        self.index_max_y = -1


        ##read csv file as dictionary: {'LIDC-IDRI-0902-2': 1, ..}
        df = pd.read_csv(target_file)
        image_targets = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))
        ##read names of all nodules without extentions
        self.available_nodules=[nodule.split('.')[0] for nodule in sorted(os.listdir(nodule_dir))]
 
        image_targets=repair_dict_names(image_targets)

        #create an array with tuples (image,label) where image and label are tensors
        self.res = []
        target_size = (64, 64) ##size of tensor
        ind=0
        indexes_overflow = [172,421,575,1846,2163]
        for x in self.available_nodules:
            nodule_path = os.path.join(self.nodule_dir, x + '.npy')
            nodule = torch.from_numpy(np.load(nodule_path))
            #extract middle slice of the nodule
            image = nodule[:, :, nodule.shape[2]//2]

            if ind in indexes_overflow or torch.allclose(image,image[0,0]):
                ind+=1
                continue

            x_size=nodule.shape[0]
            y_size=nodule.shape[1]
            z_size=nodule.shape[2]
            if x_size>self.max_x:
                self.max_x=x_size
                self.index_max_x = ind
            if y_size>self.max_y:
                self.max_y=y_size
                self.index_max_y=ind
            
            paded_image = pad_image(image, image.min(), target_size)

            label = image_targets[x] - 1 # 1-5 -> 0-4, necessary for one_hot encoding
            label = torch.tensor(label) ##dtype because of error:RuntimeError: Expected floating point type for target with class probabilities, got Long


            if self.transform:
                paded_image = self.transform(paded_image)
            if self.target_transform:
                label = self.target_transform(label, num_classes=5)

            self.res.append((paded_image, label))
            ind+=1
                # Find the maximum count among labels
            
        

    def __len__(self):
        return len(self.res)
    
    def __getitem__(self, index) -> tuple:
        return self.res[index]
    def get_max_dimensions(self):
        """
        return ((max_x,max_y),(max_x_ind,max_y,ind)) where max_x is max x dimension of nodule and max_x_ind is number of first nodule that has the max dimension
        """
        
        return ((self.max_x,self.max_y),(self.index_max_x,self.index_max_y))
    
    def get_index_name(self,index):
        return self.available_nodules[index]
    
    def get_label_by_index(self,index):
        return torch.argmax(self.res[index][1]) 

    def undersampling_majority_labels(self):
        label_counts = [0,0,0,0,0]
        for data in self.res:
            label = torch.argmax(data[1])
            label_counts[label]+=1
        print(f"Label counts in dataset: {label_counts}")

      #  res_shuffled = copy.deepcopy(self.res)
        random.shuffle(self.res)

        target_count = min(label_counts)
        label_counts_new = [0,0,0,0,0]
        balanced_data=[]

        for i in range(len(self.res)):
            label = self.get_label_by_index(i)
            if label_counts_new[label] < target_count:
                balanced_data.append(self.res[i])
                label_counts_new[label] += 1
        self.res= balanced_data

    def add_synthetic_by_rotations(self,target_count,rotations=[1,2,3]):
        first_rotated_indexes=[] #array with indexes of first data that was rotated. (used to debuging)
        label_counts = [0,0,0,0,0]
        for data in self.res:
            label = torch.argmax(data[1])
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

    ##TODO not working yet
    def oversampling_minority_labels(self):
        label_counts = [0, 0, 0, 0, 0]
        for data in self.res:
            label = torch.argmax(data[1])
            label_counts[label] += 1
        print(f"Label counts in dataset: {label_counts}")

        target_count = max(label_counts)
        label_counts_new = [0, 0, 0, 0, 0]
        balanced_data = []

        data = [item[0] for item in self.res]
        labels = [torch.argmax(item[1]) for item in self.res]

        oversample = RandomOverSampler(sampling_strategy={i: target_count for i in range(5)})
        data_resampled, labels_resampled = oversample.fit_resample(np.array(data).reshape(-1, 1), labels)

        # Convert back to the original format
        data_resampled = [torch.tensor(data_resampled[i][0]).squeeze() for i in range(len(data_resampled))]
        labels_resampled = [F.one_hot(torch.tensor(label), num_classes=5) for label in labels_resampled]

        balanced_data = list(zip(data_resampled, labels_resampled))
        self.res = balanced_data
