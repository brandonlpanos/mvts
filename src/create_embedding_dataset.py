import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from datasets import MVTSDataset
from models import TransformerEncoder
from torch.utils.data import DataLoader
from datasets import find_padding_masks

'''
In this script, we create a dataset of embeddings from the trained transformer encoder.
'''

def embedding_to_dataframe(embedding, features, label, indx):
    """
    This function creates a pandas dataframe of the embeddings from transformer encoder that has been trained on the classifcation taks.
    The dataframe is of the same format as the original csv files.
    Args: embedding --> from trained transformer encoder ndarray(seq_len, d_model)
          features --> list of features from original csv file
          label --> Flare No-Flare binary label output from the dataloader 
          indx --> index of the original csv file         
    Output: pandas dataframe with the same format as the original csv files.
    """

    embedding = np.concatenate(
        (np.full((embedding.shape[0], 1), indx, dtype=int), embedding), axis=1) # add index column

    # add extra columns to match original csv file
    extra_cols = len(features) - embedding.shape[1] 
    extra_data = np.empty((embedding.shape[0], extra_cols))
    extra_data.fill(np.nan)
    modified_array = np.concatenate((embedding, extra_data), axis=1)

    # turn array into pandas dataframe and rename columns as well as add target values
    new_df = pd.DataFrame(modified_array, columns=features)
    new_df['R_VALUE'], new_df['XR_MAX'] = new_df['XR_MAX'], new_df['R_VALUE']
    new_df['target'] = label

    return new_df

# Get column names
df = pd.read_csv('../data/long/1.csv') # read in random file since all csv files have the same columns
features = df.columns.tolist()

# Load all data
indices = np.arange(0, 485, 1)
dataloader = DataLoader(MVTSDataset(indices, norm_type='standard'), batch_size=len(indices), shuffle=False, drop_last=False)
data, _, y = next(iter(dataloader))

# Create the embeddings using the trained transformer model
transformer_model = TransformerEncoder(feat_dim=35,
                                       max_len=40,
                                       d_model=35,
                                       n_heads=7,
                                       num_layers=1,
                                       dim_feedforward=256,
                                       dropout=0.1,
                                       freeze=True).float()
transformer_model.eval();

indx = 0
for input, label in zip(data, y):
    label = label.item()
    input = input.unsqueeze(0)
    padding_mask = find_padding_masks(input)
    input = torch.nan_to_num(input)
    output, embedding = transformer_model(input, padding_mask)
    embedding = embedding.detach().numpy()[0]
    padding_mask = padding_mask.squeeze(0).detach().numpy()
    embedding = embedding[padding_mask]
    df = embedding_to_dataframe(embedding, features, label, indx)
    df['Unnamed: 0'] = df['Unnamed: 0'].astype(int)
    df.rename(columns={'Unnamed: 0': ''}, inplace=True)
    df.to_csv(f'../data/embeddings/{indx}.csv', index=False)
    indx += 1