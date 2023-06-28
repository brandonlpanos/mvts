import torch
import numpy as np

#? Function for first order topological shuffle (shuffle across time acess for each feature)
def shuffle_tensor_along_time(tensor):
    # Runs dynamicaly on during traning loop on batches. Input shape (batch_size, time_steps, d_features)
    batch_size, time_steps, d_features = tensor.size()
    indices = torch.stack([torch.randperm(time_steps) for _ in range(batch_size * d_features)]).view(batch_size, d_features, time_steps)
    shuffled_tensor = tensor.permute(0, 2, 1).gather(2, indices).permute(0, 2, 1)
    return shuffled_tensor

#? Function for second order topological shuffle (shuffle across time and feature access)
def topological_shuffle(tensor):
    # Runs dynamicaly on during traning loop on batches. Input shape (batch_size, time_steps, d_features)
    batch_size, time_steps, d_features = tensor.size()
    shuffled_tensor = tensor.clone()
    for i in range(batch_size):
        indices = torch.randperm(time_steps * d_features)
        shuffled_tensor[i] = tensor[i].view(-1)[indices].view(time_steps, d_features)
    return shuffled_tensor

#? Integrates out the intensity by normalizing each feature from a single mvts by its maximum value
def unity_based_normalization(data):
    # Applied implicitly in the dataloader but can be dynamicly run on single instances if batch size is 1: input shape (time_steps, d_features)
    max_vals = np.nanmax(data, axis=1)
    min_vals = np.nanmin(data, axis=1)
    ranges = max_vals - min_vals
    eps = np.finfo(data.dtype).eps  
    ranges[ranges < eps] = eps
    data = (data - min_vals[:, np.newaxis]) / ranges[:, np.newaxis]
    data = data + np.nanmax(data)
    data *= (1 / np.nanmax(data, axis=1)[:, np.newaxis])
    return data

#? Identity normalization
def identity_normalization(tensor):
    return tensor

#? Robust Standardization
# Applied implicitly in the dataloader and cannot be run dynamicaly on single batchs