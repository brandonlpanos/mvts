import json
import torch
import torch.nn as nn
from datasets import ImputationDataset
from datasets import find_padding_masks
from torch.utils.data import DataLoader
from models import TransformerEncoder, CNNModel, CombinedModel


def train_and_validate_classifier(model, train_loader, test_loader, n_epoch):

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for epoch in range(n_epoch):

        # Training phase
        train_loss = 0
        model.train()
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)

        # Validation phase
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)

        train_loss /= len(train_loader)
        val_loss /= len(test_loader)

    return 


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

# Initiate the CNN model
cnn_model = CNNModel()

# Combine both models
main_model = CombinedModel(transformer_model, cnn_model)

# Create training loops