import json
import torch
import torch.nn as nn
from datasets import ImputationDataset
from datasets import find_padding_masks
from torch.utils.data import DataLoader
from models import TransformerEncoder, CNNModel, CombinedModel


def train_and_validate_classifier(model, train_loader, test_loader, n_epoch, save_path):

    best_test_loss = 1e20
    running_batch_loss_train = []
    running_batch_loss_test = []

    for epoch in range(n_epoch):
        print(f"Epoch: {epoch + 1}")

        # Train loop 
        for x, _, y in train_loader:
            padding_mask = find_padding_masks(x)
            x = torch.nan_to_num(x).to(device)
            y_hat = main_model(x, padding_mask).to(device)
            optimizer.zero_grad()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            running_batch_loss_train.append(loss.item())

        # Test loop
        for x, _, y in test_loader:
            padding_mask = find_padding_masks(x)
            x = torch.nan_to_num(x).to(device)
            y_hat = main_model(x, padding_mask).to(device)
            loss = criterion(y_hat, y)
            running_batch_loss_test.append(loss.item())

        # Save model if test loss is lower than best test loss
        if running_batch_loss_test[-1] < best_test_loss:
            best_test_loss = running_batch_loss_test[-1]
            torch.save(main_model.state_dict(), save_path)
            print(f"Saved model at epoch {epoch + 1}")

        # Print loss
        print(f"Train loss: {running_batch_loss_train[-1]}")
        print(f"Test loss: {running_batch_loss_test[-1]}")

    return running_batch_loss_train, running_batch_loss_test


if __name__ == '__main__':

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
    cnn_model = CNNModel().float()

    # Combine both models
    main_model = CombinedModel(transformer_model, cnn_model).float()

    # Load the dataset
    with open('../data/data_indices.json', 'r') as f: data_indices = json.load(f)
    train_indices = data_indices['train_indices']
    train_dataloader = DataLoader(ImputationDataset(train_indices, norm_type='unity', mean_mask_length=3, masking_ratio=0.15), batch_size=10, shuffle=True, drop_last=True)
    test_indices = data_indices['test_indices']
    test_dataloader = DataLoader(ImputationDataset(test_indices, norm_type='unity', mean_mask_length=3, masking_ratio=0.15), batch_size=10, shuffle=True, drop_last=True)

    # Train and validate the model
    n_epoch = 100
    train_loader = train_dataloader
    criterion = nn.CrossEntropyLoss()
    save_path = '../models/classification_unity_norm.pt'
    optimizer = torch.optim.Adam(main_model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_model.to(device)
    train_loss, test_loss = train_and_validate_classifier(main_model, train_dataloader, test_dataloader, n_epoch, save_path)