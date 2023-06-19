import json
import torch
import torch.nn as nn
from datasets import MVTSDataset
from torch.utils.data import DataLoader
from models import CNNModel

# Define train and test loops 

def train():
    model.train()
    train_loss = 0
    train_correct = 0
    total_samples = 0  # Variable to keep track of total samples
    for i, (x, y) in enumerate(train_dataloader):
        x = x.to(device)
        y = y.to(device).long()  # Convert the target tensor to long
        optimizer.zero_grad()
        probabilities = model(x)
        loss = criterion(probabilities, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (probabilities.argmax(dim=-1) == y).sum().item()
        total_samples += x.size(0)  # Increment the total samples by batch size
    return train_loss / len(train_dataloader), train_correct / total_samples  # Divide by total_samples

def val():
    model.eval()
    val_loss = 0
    val_correct = 0
    total_samples = 0  # Variable to keep track of total samples
    for i, (x, y) in enumerate(val_dataloader):
        x = x.to(device)
        y = y.to(device).long()  # Convert the target tensor to long
        probabilities = model(x)
        loss = criterion(probabilities, y)
        val_loss += loss.item()
        val_correct += (probabilities.argmax(dim=-1) == y).sum().item()
        total_samples += x.size(0)  # Increment the total samples by batch size
    return val_loss / len(val_dataloader), val_correct / total_samples  # Divide by total_samples


if __name__ == 'Main'