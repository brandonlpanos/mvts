import json
import torch
import torch.nn as nn
from models import TransformerEncoder
from datasets import ImputationDataset
from datasets import find_padding_masks
from torch.utils.data import DataLoader

# Initiate the transformer model
transformer_model = TransformerEncoderInputter(feat_dim=35,
                                    max_len=40,
                                    d_model=64, 
                                    n_heads=8, 
                                    num_layers=1,
                                    dim_feedforward=256, 
                                    dropout=0.1, 
                                    freeze=False)
transformer_model.float()

# Load pretrained weights
transformer_model.load_state_dict(torch.load('../models/inputting_unity_norm.pt'))




