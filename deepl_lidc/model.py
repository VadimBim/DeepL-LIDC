from deepl_lidc import dataset,patient_info
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.model_selection import train_test_split
import numpy as np
import os



# Path to the data directory
PATH_scans = 'C:\Pliki\sem5\Advanced_data_mining\.Vadim\DeepL-LIDC\data\.nodules'
# Path to the target data directory
PATH_target = 'C:\Pliki\sem5\Advanced_data_mining\.Vadim\DEEPL-LIDC\data\.nodule_target.csv'
PATH_results = r'C:\Pliki\sem5\Advanced_data_mining\.Vadim\DEEPL-LIDC\results\Training_model'
os.makedirs(PATH_results, exist_ok=True)

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
#patient_info.plot_images_by_indexes(lidc_dataset,indexes_flips)

# Split data into training, validation, and test sets
train_size = 0.8
val_size = 0.1
test_size = 1 - train_size - val_size

train_indices, temp_indices = train_test_split(list(range(len(lidc_dataset))), test_size=val_size + test_size, random_state=42)
val_indices, test_indices = train_test_split(temp_indices, test_size=test_size/(val_size + test_size), random_state=42)

train_dataset = torch.utils.data.Subset(lidc_dataset, train_indices)
val_dataset = torch.utils.data.Subset(lidc_dataset, val_indices)
test_dataset = torch.utils.data.Subset(lidc_dataset, test_indices)

# DataLoaders for training, validation, and test sets
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



learning_rates = [0.1, 0.01, 0.001,0.0001,0.00001]
train_losses = {lr: [] for lr in learning_rates}
val_losses = {lr: [] for lr in learning_rates}
accuracies = {lr: [] for lr in learning_rates}


for model_id, LR in enumerate(learning_rates, start=1):
  print(f"Training with learning_rate: {LR}")
  model = Net()
  ## loss function and optimizer:
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr=LR)

  num_epochs=51

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
        # Save training history
      train_losses[LR].append(loss.item())
      val_losses[LR].append(average_val_loss)
      accuracies[LR].append(accuracy)

  # Testing loop
  model.eval()
  test_loss = 0.0
  correct = 0
  total = 0
  with torch.no_grad():
      for batch in test_loader:
          inputs, targets = batch
          outputs = model(inputs)
          loss = torch.nn.functional.cross_entropy(outputs, targets.float())
          test_loss += loss.item()
          _, predicted = outputs.max(1)
          total += targets.size(0)
          _, expected_labels = targets.max(1)
          for k in range(targets.size(0)):
              if predicted[k] == expected_labels[k]:
                  correct += 1

  average_test_loss = test_loss / len(test_loader)
  accuracy = correct / total

  print(f'Test Loss: {average_test_loss:.4f}, '
        f'Test Accuracy: {accuracy * 100:.2f}%')
        # Save training history
  # Save the model's state dictionary
  model_state_path = os.path.join(PATH_results, f'Model_{model_id}_state.pt')
  torch.save(model.state_dict(), model_state_path)

for model_id, LR in enumerate(learning_rates, start=1):
     # Save the training history plot
    plt.figure()
    plt.plot(train_losses[LR], label='Training Loss')
    plt.plot(val_losses[LR], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.xticks([z for z in range(0,num_epochs,2)])
    plt.ylabel('Loss')
    plt.legend()
    plt.title(f'Training History (Model with LR={LR})')
    plt.savefig(os.path.join(PATH_results, f'Training_history_Model_LR_{model_id}.png'))
    plt.close()

    # Save the accuracy plot
    plt.figure()
    plt.plot(accuracies[LR], label='Accuracy')
    plt.xlabel('Epoch')
    plt.xticks([z for z in range(0,num_epochs,2)])
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(f'Accuracy History (Model with LR={LR})')
    plt.savefig(os.path.join(PATH_results, f'Accuracy_Model_{model_id}.png'))
    plt.close()
    
plt.figure()
for LR in learning_rates:
    #Accuracy for different LR:
    plt.plot(accuracies[LR], label=f'Learning Rate: {LR}')
# Save the global accuracy plot
plt.xlabel('Epoch')
plt.xticks([z for z in range(0,num_epochs,2)])
plt.ylabel('Accuracy')
plt.legend()
plt.title('Accuracy for Different Learning Rates')
plt.savefig(os.path.join(PATH_results, 'accuracy_for_different_learning_rates.png'))
plt.close()