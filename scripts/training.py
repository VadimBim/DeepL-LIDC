"""Train the model."""
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from deepl_lidc.dataset import NoduleDataset
from deepl_lidc.models import NeuralNetwork
from deepl_lidc.train import train_loop, test_loop

LEARNING_RATE = 0.001
BATCH_SIZE = 14
EPOCHS = 100

target_file = "/home/vadim/Development/Projects/DeepL-LIDC/data/nodule_target.csv"
nodule_dir = "/home/vadim/Development/Projects/DeepL-LIDC/data/nodules"
model_path = "/home/vadim/Development/Projects/DeepL-LIDC/results/model.pth"

dataset = NoduleDataset(target_file, nodule_dir)
total_length = len(dataset)
train_length = int(0.8 * total_length)
test_length = total_length - train_length

train_set, test_set = random_split(dataset, [train_length, test_length])

train_dataloader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

model = NeuralNetwork()

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

if __name__ == "__main__":
    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train_loop(train_dataloader, model, loss_fn, optimizer)
        test_loop(test_dataloader, model, loss_fn)
    print("Done!")
    torch.save(model.state_dict(), model_path)