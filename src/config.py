import torch
import numpy as np
from normalizations import shuffle_tensor_along_time, topological_shuffle, identity_normalization

N_EPOCHS = 50 # Number of epochs to train models for the binary classification task
N_EPOCHS_AR = 200 # Number of epochs to train models for the autoregressive denoising task
BASE_NORM = 'unity' # Normalization type for the binary classification task| Options: 'unity', 'robust'
ACTIVE_NORM = shuffle_tensor_along_time # Normalization type for the autoregressive denoising task | Options: shuffle_tensor_along_time, topological_shuffle, identity_normalization
