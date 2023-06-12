import os
import json
import torch
import numpy as np
from losses import MaskedMSELoss
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from matplotlib.ticker import MultipleLocator 
from models import TransformerEncoderInputter
from datasets import ImputationDataset, find_padding_masks


def train_and_validate_inputer(model, train_loader, test_loader, n_epoch):

    best_test_loss = 1e20
    running_batch_loss_train = []
    running_batch_loss_test = []
    save_path = '../models/inputting_unity_norm.pt'
    if 'inputting_unity_norm.pt' in os.listdir(os.path.dirname(save_path)):os.remove(save_path)

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

            y_hat = model(x_masked, padding_mask)
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
      
                y_hat = model(x_masked, padding_mask)
                loss = critereon(y_hat, x, target_masks)
                running_batch_loss_test.append(loss.item())

            if loss < best_test_loss:
                best_test_loss = loss
                torch.save(model.state_dict(), save_path)

        print(f"Epoch Train Loss: {np.nanmean(running_batch_loss_train)}\nEpoch Test Loss: {np.nanmean(running_batch_loss_test)}")
                
    return running_batch_loss_train, running_batch_loss_test

if __name__ == "__"