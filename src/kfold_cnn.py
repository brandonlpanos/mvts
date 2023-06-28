import torch
import config
import numpy as np
import torch.nn as nn
from models import CNNModel
from datasets import MVTSDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

'''
This script is used to train a CNN model using k-fold cross-validation.
'''

def train():
    model.train()
    train_loss = 0
    train_correct = 0
    total_samples = 0  # Variable to keep track of total samples
    for i, (x, mask, y) in enumerate(train_dataloader):
        x = torch.nan_to_num(x).to(device).unsqueeze(1)  # Add a channel dimension
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
    for i, (x, mask, y) in enumerate(val_dataloader):
        x = torch.nan_to_num(x).to(device).unsqueeze(1)  # Add a channel dimension
        y = y.to(device).long()  # Convert the target tensor to long
        probabilities = model(x)
        loss = criterion(probabilities, y)
        val_loss += loss.item()
        val_correct += (probabilities.argmax(dim=-1) == y).sum().item()
        total_samples += x.size(0)  # Increment the total samples by batch size
    return val_loss / len(val_dataloader), val_correct / total_samples  # Divide by total_samples



if __name__ == '__main__':

    # Collect filenames for all 50 models
    path_to_splits = '../kfold/splits/'
    file_names = np.arange(0, 50, 1)
    file_names = np.delete(file_names, np.argwhere(file_names == 27))
    file_names = np.delete(file_names, np.argwhere(file_names == 37))

    for file_name in file_names:

        # Load split indices
        path = path_to_splits + 'fold_' + str(file_name) + '.npz'
        fhand = np.load(path)
        val_indices = fhand['val_indices']
        train_indices = fhand['train_indices']

        # Create dataloaders
        val_dataloader = DataLoader(MVTSDataset(val_indices, norm_type='unity'), batch_size=16, shuffle=True, drop_last=True)
        train_dataloader = DataLoader(MVTSDataset(train_indices, norm_type='unity'), batch_size=16, shuffle=True, drop_last=True)

        # Define model, optimizer, and loss function
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        criterion = nn.CrossEntropyLoss()
        model = CNNModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Train and validate the model
        best_loss = np.inf
        best_val_acc = 0
        best_val_loss = float('inf')
        best_model_state_dict = None

        for epoch in range(config.N_EPOCHS):
            train_loss, train_acc = train()
            val_loss, val_acc = val()
            
            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = model.state_dict()
                best_val_acc = val_acc
            
                print(f'Epoch {epoch + 1} | Train Loss: {train_loss:.5f} | Train Acc: {train_acc * 100:.2f}% | Val Loss: {val_loss:.5f} | Val Acc: {val_acc * 100:.2f}%')


        # Save the best model to a file
        torch.save(best_model_state_dict, f'../kfold/models_cnn_unity/{file_name}.pth')

        # Clean up memory
        del model, optimizer, criterion, train_dataloader, val_dataloader, best_model_state_dict