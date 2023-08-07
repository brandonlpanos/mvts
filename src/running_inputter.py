import os
import json
import torch
import config
import numpy as np
from losses import MaskedMSELoss
from torch.utils.data import DataLoader
from models import TransformerEncoder
from datasets import MVTSDataset, find_padding_masks


def train_and_validate_inputer(model, train_loader, test_loader, n_epoch, save_path):
    """
    Trains and validates a model using input data.

    Args:
        model (torch.nn.Module): The model to be trained and validated.
        train_loader (torch.utils.data.DataLoader): The data loader for the training set.
        test_loader (torch.utils.data.DataLoader): The data loader for the test/validation set.
        n_epoch (int): The number of epochs to train the model.

    Returns:
        Tuple of two lists: 'running_batch_loss_train' and 'running_batch_loss_test'.
        'running_batch_loss_train' contains the training loss for each batch in each epoch.
        'running_batch_loss_test' contains the test loss for each batch in each epoch.

    The traning loop saves the model with the lowest test loss to '../models/inputting_unity_norm.pt'.
    
    Autoregressive denoising requres several maskes, filters and dtypses:
    1) traget = (batch, max_seq_len, feat_dim) tensor with missing values at the beginning replaced by 0
    2) mask = (batch, max_seq_len, feat_dim) boolean tensor
    3) padding_mask = (batch, max_seq_len) boolean tensor, 1 means keep, 0 means ignore dtype=torch.bool
    4) x_masked = (batch, max_seq_len, feat_dim) tensor with missing values that follow a geometric distribution
    5) target_masks = (batch, max_seq_len, feat_dim) boolean tensor 1 means area to calculate the loss on, opposite to mask
    """

    best_test_loss = 1e20
    running_batch_loss_train = []
    running_batch_loss_test = []

    for epoch in range(n_epoch):
        print(epoch + 1)

        model.train()
        for x, mask, _ in train_loader:
            padding_mask = find_padding_masks(x) # shape (batchsize, max_seq_len) to pad start of short sequences
            x = torch.nan_to_num(x) # replace nan with 0 (since needs to be processed by the model)
            x_masked = x * mask # mask the input
            target_masks = ~mask  # inverse logic: 0 now means ignore, 1 means predict (when passed to the loss function)

            x = x.to(device)
            x_masked = x_masked.to(device)
            target_masks = target_masks.to(device)
            padding_mask = padding_mask.to(device)

            y_hat, embeddings = model(x_masked, padding_mask)
            loss = critereon(y_hat, x, target_masks)
            optimizer.zero_grad()
            loss.backward() 
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=4.0)
            optimizer.step()
            running_batch_loss_train.append(loss.item())

            model.eval()
            with torch.no_grad():
                x, mask, _ = next(iter(test_loader))
                padding_mask = find_padding_masks(x) 
                x = torch.nan_to_num(x)
                x_masked = x * mask 
                target_masks = ~mask 

                x = x.to(device)
                x_masked = x_masked.to(device)
                target_masks = target_masks.to(device)
                padding_mask = padding_mask.to(device)
      
                y_hat, embeddings = model(x_masked, padding_mask)
                loss = critereon(y_hat, x, target_masks)
                running_batch_loss_test.append(loss.item())

                if loss < best_test_loss:
                    best_test_loss = loss
                    torch.save(model.state_dict(), save_path)

        print(f"Epoch Train Loss: {np.nanmean(running_batch_loss_train)}\nEpoch Test Loss: {np.nanmean(running_batch_loss_test)}")
                
    return running_batch_loss_train, running_batch_loss_test

if __name__ == "__main__":

    # Create model
    # dropout = 0.1 dropout
    # n_heads = 8 number of heads
    # num_layers = 1 number of layers
    # feat_dim = 35 number of features
    # max_len = 40 max length of sequence
    # d_model = 64 dimension of the model
    # freeze = False freeze the model True --> no dropout
    # dim_feedforward = 256 dimension of the feedforward layers within the transformer blocks
    model = TransformerEncoder(feat_dim=35,
                               max_len=40,
                               d_model=35, 
                               n_heads=7, 
                               num_layers=1,
                               dim_feedforward=256, 
                               dropout=0.1, 
                               freeze=False)

    # Create an instance of the model and set to float (default is double)
    model.float();

    # Read in train and validation indices
    with open('../data/data_indices.json', 'r') as f: data_indices = json.load(f)
    train_indices = data_indices['train_indices']
    val_indices = data_indices['val_indices']

    # Create loaders
    train_dataloader = DataLoader(MVTSDataset(train_indices, norm_type='standard', mean_mask_length=3, masking_ratio=0.15), batch_size=10, shuffle=True, drop_last=True)
    val_dataloader = DataLoader(MVTSDataset(val_indices, norm_type='standard', mean_mask_length=3, masking_ratio=0.15), batch_size=10, shuffle=True, drop_last=True)

    # Set device, initiate optimizer, define loss criterion, and set number of epoch. Finally, train and validate the model
    critereon = MaskedMSELoss()
    save_path = '../models/inputting_combined_std.pt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    running_batch_loss_train, running_batch_loss_test = train_and_validate_inputer(model,
                                                                                   train_dataloader,
                                                                                   val_dataloader,
                                                                                   config.N_EPOCHS_AR, 
                                                                                   save_path
    )
    np.savez(f'../models/training_curves_inputting_standard.npz', loss_train=running_batch_loss_train, loss_test=running_batch_loss_test)