import os
import sys
import torch
sys.path.insert(0, "../../src")
import config
import numpy as np
import torch.nn as nn
from models import CNNModel
from normalizations import *
from datasets import MVTSDataset
from torch.utils.data import DataLoader
from datasets import find_padding_masks
from models import TransformerEncoder, CNNModel, CombinedModel

"""
This script is used to train the combined model over 50 random train/val splits of the dataset.
"""

# Train and validation loops for the combined model

def train():
    combined_model.train()
    train_loss = 0
    train_correct = 0
    total_samples = 0  # Variable to keep track of total samples
    for i, (x, mask, y) in enumerate(train_dataloader):
        padding_mask = find_padding_masks(x).to(device)
        x = config.ACTIVE_NORM(x) #? Secondary layer of normalization
        x = torch.nan_to_num(x).to(device)
        y = y.to(device).long()  # Convert the target tensor to long
        optimizer.zero_grad()
        probabilities = combined_model(x, padding_mask).to(device)
        loss = criterion(probabilities, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_correct += (probabilities.argmax(dim=-1) == y).sum().item()
        total_samples += x.size(0)  # Increment the total samples by batch size
    return train_loss / len(train_dataloader), train_correct / total_samples  # Divide by total_samples

def val():
    combined_model.eval()
    val_loss = 0
    val_correct = 0
    total_samples = 0  # Variable to keep track of total samples
    for i, (x, mask, y) in enumerate(val_dataloader):
        padding_mask = find_padding_masks(x).to(device)
        x = config.ACTIVE_NORM(x) #? Secondary layer of normalization
        x = torch.nan_to_num(x).to(device)
        y = y.to(device).long()  # Convert the target tensor to long
        probabilities = combined_model(x, padding_mask).to(device)
        loss = criterion(probabilities, y)
        val_loss += loss.item()
        val_correct += (probabilities.argmax(dim=-1) == y).sum().item()
        total_samples += x.size(0)  # Increment the total samples by batch size
    return val_loss / len(val_dataloader), val_correct / total_samples  # Divide by total_samples


if __name__ == "__main__":
    
    # Collect filenames for all 50 models
    path_to_splits = '../../kfold_results/splits/'
    file_names = np.arange(0, 50, 1)
    file_names = np.delete(file_names, np.argwhere(file_names == 27))
    file_names = np.delete(file_names, np.argwhere(file_names == 37))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Loop over random splits
    for file_name in file_names:

        # Load split indices
        path = path_to_splits + 'fold_' + str(file_name) + '.npz'
        fhand = np.load(path)
        val_indices = fhand['val_indices']
        train_indices = fhand['train_indices']

        # Create dataloaders
        val_dataloader = DataLoader(MVTSDataset(val_indices, norm_type=config.BASE_NORM), batch_size=16, shuffle=True, drop_last=True)
        train_dataloader = DataLoader(MVTSDataset(train_indices, norm_type=config.BASE_NORM), batch_size=16, shuffle=True, drop_last=True)

        # Initiate combined model
        transformer_model = TransformerEncoder(feat_dim=35,
                                                max_len=40,
                                                d_model=35, 
                                                n_heads=7, 
                                                num_layers=1,
                                                dim_feedforward=256, 
                                                dropout=0.1, 
                                                freeze=False)

        transformer_model.float()
        cnn_model = CNNModel().float()
        combined_model = CombinedModel(transformer_model, cnn_model).float().to(device)

        # Define optimizer, and loss function
        optimizer = torch.optim.Adam(combined_model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        # Initiate best model parameters
        best_loss = np.inf
        best_val_acc = 0
        best_val_loss = float('inf')
        best_model_state_dict = None

        # Training loop for each split
        for epoch in range(config.N_EPOCHS):

            train_loss, train_acc = train()
            val_loss, val_acc = val()
            
            # Save the best model based on validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = combined_model.state_dict()
                best_val_acc = val_acc
            
                print(f'Epoch {epoch + 1} | Train Loss: {train_loss:.5f} | Train Acc: {train_acc * 100:.2f}% | Val Loss: {val_loss:.5f} | Val Acc: {val_acc * 100:.2f}%')

        # Save the best model to a file
        base_save_dir = '../../models/' + config.RUN_NAME + '/'
        if os.path.exists(base_save_dir) == False:
            os.mkdir(base_save_dir)
        torch.save(best_model_state_dict, f'{base_save_dir}{file_name}.pth')

        # Clean up memory
        del transformer_model, cnn_model, combined_model, optimizer, criterion, train_dataloader, val_dataloader, best_model_state_dict