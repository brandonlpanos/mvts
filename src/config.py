import torch
import numpy as np
from normalizations import shuffle_tensor_along_time, topological_shuffle, identity_normalization

N_EPOCHS = 50 # Number of epochs to train models for the binary classification task
N_EPOCHS_AR = 200 # Number of epochs to train models for the autoregressive denoising task
BASE_NORM = 'standard' # Normalization type for the binary classification task| Options: 'unity', 'standard'
ACTIVE_NORM = identity_normalization # Normalization type for the autoregressive denoising task | Options: shuffle_tensor_along_time, topological_shuffle, identity_normalization
RUN_NAME = 'cnn_std' # Name of the run