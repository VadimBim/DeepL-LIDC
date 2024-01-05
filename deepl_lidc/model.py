from deepl_lidc import dataset,patient_info
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np


# Path to the data directory
PATH_scans = 'C:\Pliki\sem5\Advanced_data_mining\.Vadim\DeepL-LIDC\data\.nodules'
# Path to the target data directory
PATH_target = 'C:\Pliki\sem5\Advanced_data_mining\.Vadim\DEEPL-LIDC\data\.nodule_target.csv'


torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(42)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, 5)  # 5 klas (0-4)

    def forward(self, x):
        x = F.relu(self.conv1(x.unsqueeze(1)))  # delete dimension
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 64 * 16 * 16)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

lidc_dataset = dataset.NoduleDataset(PATH_target,PATH_scans)
#lidc_dataset.undersampling_majority_labels()
#lidc_dataset.oversampling_minority_labels()
indexes_rotated = lidc_dataset.add_synthetic_by_rotations(700)
indexes_flips = lidc_dataset.add_synthetic_by_flips(1006)

#patient_info.plot_images_by_indexes(lidc_dataset,indexes_rotated)
patient_info.plot_images_by_indexes(lidc_dataset,indexes_flips)

##split data:
train_size = 0.8
train_indices, val_indices = train_test_split(list(range(len(lidc_dataset))), test_size=1 - train_size, random_state=42)
train_dataset = torch.utils.data.Subset(lidc_dataset, train_indices)
val_dataset = torch.utils.data.Subset(lidc_dataset, val_indices)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

model = Net()

## loss function and optimizer:
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)



num_epochs=35
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        inputs, targets = batch
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = torch.nn.functional.cross_entropy(outputs, targets.float())
        loss.backward()
        optimizer.step()

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            outputs = model(inputs)
            loss = torch.nn.functional.cross_entropy(outputs, targets.float())
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            _, expected_labels = targets.max(1)
            for k in range(targets.size(0)):
             
                if predicted[k]==expected_labels[k]:
                    correct+=1

    average_val_loss = val_loss / len(val_loader)
    accuracy = correct / total

    print(f'Epoch [{epoch + 1}/{num_epochs}], '
          f'Loss: {loss.item():.4f}, '
          f'Validation Loss: {average_val_loss:.4f}, '
          f'Accuracy: {accuracy * 100:.2f}%')



