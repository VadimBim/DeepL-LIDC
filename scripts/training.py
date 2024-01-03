"""Train the model."""
import torch
from torch import nn
from torch.utils.data import random_split, DataLoader
from deepl_lidc.dataset import NoduleDataset
from deepl_lidc.models import NeuralNetwork

LEARNING_RATE = 0.001
BATCH_SIZE = 28
EPOCHS = 100

target_file = "/home/vadim/Development/Projects/DeepL-LIDC/data/nodule_target.csv"
nodule_dir = "/home/vadim/Development/Projects/DeepL-LIDC/data/nodules"

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

def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    #Set model to train mode - important for dropout and batchnorm
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test_loop(dataloader, model, loss_fn):
    #Set model to eval mode - important for dropout and batchnorm 
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
for t in range(EPOCHS):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")