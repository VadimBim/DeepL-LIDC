""" 
Module for training the model an evaluating it on the test set.
"""
import torch

def train_loop(dataloader, model, loss_fn, optimizer):
    """Train the model on the training set.

    Args:
        dataloader (torch.DataLoader): The training dataloader.
        model (nn.Module): The model to train.  
        loss_fn (nn.loss_function): The loss function to use.
        optimizer (nn.optimizer): The optimizer to use.
    """
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
        
        if batch % 14 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test_loop(dataloader, model, loss_fn):
    """Evaluate performance of the trained model on the test set.

    Args:
        dataloader (torch.DataLoader): The test dataloader.
        model (nn.Module): The trained model.
        loss_fn (nn.loss_function): The loss function to use.
    """
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