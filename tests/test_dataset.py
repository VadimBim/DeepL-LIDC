import torch
import numpy as np
import pytest
from deepl_lidc.dataset import pad_image, NoduleDataset
import matplotlib.pyplot as plt

def test_pad_image():
    # Test case 1: 2D tensor
    tensor = torch.tensor([[1, 2], [3, 4]])
    padding_value = 0
    target_size = (4, 4)
    expected_output = torch.tensor([[0, 0, 0, 0], [0, 1, 2, 0], [0, 3, 4, 0], [0, 0, 0, 0]])
    assert torch.allclose(pad_image(tensor, padding_value, target_size), expected_output)
    
    #test case 3x3 to 5x5
    tensor = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    padding_value = 0
    target_size = (5, 5)
    expected_output = torch.tensor([[0, 0, 0, 0, 0], [0, 1, 2, 3, 0], [0, 4, 5, 6, 0], [0, 7, 8, 9, 0], [0, 0, 0, 0, 0]])
    assert torch.allclose(pad_image(tensor, padding_value, target_size), expected_output)
    
    #test for a image
    nodule = torch.from_numpy(np.load('/home/vadim/Development/Projects/DeepL-LIDC/data/nodules/LIDC-IDRI-0001-1.npy'))
    image = nodule[:, :, nodule.shape[2] // 2]
    padding_value = 0
    target_size = (84, 84)
    expected_shape = torch.Size([84, 84])
    assert pad_image(image, padding_value, target_size).shape == expected_shape

def test_nodule_dataset():
    # Test case 1: Check if the dataset returns the correct length
    target_file = "/home/vadim/Development/Projects/DeepL-LIDC/data/nodule_target.csv"
    nodule_dir = "/home/vadim/Development/Projects/DeepL-LIDC/data/nodules"
    dataset = NoduleDataset(target_file, nodule_dir)
    assert len(dataset) == 2616
    
    #Check if there are no images if nan
    for i in range(len(dataset)):
        assert not torch.isnan(dataset[i][0]).any()
        
    #Check if all images are 84x84
    for i in range(len(dataset)):
        assert dataset[i][0].shape == torch.Size([84, 84])
    
    # Test case 2: Check if the dataset returns the correct sample
    sample_idx = 0
    image, label = dataset[sample_idx]
    expected_label = torch.tensor([0., 0., 0., 0., 1.0])
    assert image.shape == torch.Size([84, 84])
    assert torch.allclose(label, expected_label)
    
    # Test case 3: Iterate and visualize the first 9 samples
    labels_map = {
        1: 'Highly Unlikely',
        2: 'Moderately Unlikely',
        3: 'Indeterminate',
        4: 'Moderately Suspicious',
        5: 'Highly Suspicious'
    }
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3
    for i in range(1, cols * rows + 1):
        image, label = dataset[i - 1]
        figure.add_subplot(rows, cols, i)
        plt.title(labels_map[label.argmax().item() + 1])
        plt.axis("off")
        plt.imshow(image, cmap="gray")
    figure.suptitle(f"First 9 samples of the dataset.")
    # plt.savefig('/home/vadim/Development/Projects/DeepL-LIDC/results/nodules.png', dpi=200)
    plt.show()