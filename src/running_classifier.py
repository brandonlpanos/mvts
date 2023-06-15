import json
import torch
import torch.nn as nn
from datasets import MVTSDataset
from datasets import find_padding_masks
from torch.utils.data import DataLoader
from models import TransformerEncoder, CNNModel, CombinedModel


def train_and_validate_classifier(model, train_loader, test_loader, n_epoch, save_path):
    
    best_test_loss = 1e20
    running_batch_loss_train = []
    running_batch_loss_test = []
    running_batch_accuracy_train = []
    running_batch_accuracy_test = []

    for epoch in range(n_epoch):
        print(f"Epoch: {epoch + 1}")

        # Train loop
        model.train()
        correct_train = 0
        total_train = 0
        train_loss = 0

        for x, _, y in train_loader:
            padding_mask = find_padding_masks(x)
            x = torch.nan_to_num(x).to(device)
            y_hat = model(x, padding_mask).to(device)

            optimizer.zero_grad()
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            _, predicted_train = torch.max(y_hat.data, 1)
            total_train += y.size(0)
            correct_train += (predicted_train == y).sum().item()

        running_batch_loss_train.append(train_loss / len(train_loader))
        running_batch_accuracy_train.append(100 * correct_train / total_train)

        # Test loop
        model.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0

        for x, _, y in test_loader:
            padding_mask = find_padding_masks(x)
            x = torch.nan_to_num(x).to(device)
            y_hat = model(x, padding_mask).to(device)

            loss = criterion(y_hat, y)
            test_loss += loss.item()

            _, predicted_test = torch.max(y_hat.data, 1)
            total_test += y.size(0)
            correct_test += (predicted_test == y).sum().item()

        running_batch_loss_test.append(test_loss / len(test_loader))
        running_batch_accuracy_test.append(100 * correct_test / total_test)

        # Save model if test loss is lower than best test loss
        if running_batch_loss_test[-1] < best_test_loss:
            best_test_loss = running_batch_loss_test[-1]
            torch.save(model.state_dict(), save_path)
            print(f"Saved model at epoch {epoch + 1}")

        # Print loss and accuracy
        print(f"Train loss: {running_batch_loss_train[-1]:.4f}, Train Accuracy: {running_batch_accuracy_train[-1]:.2f}%")
        print(f"Test loss: {running_batch_loss_test[-1]:.4f}, Test Accuracy: {running_batch_accuracy_test[-1]:.2f}%")

    return running_batch_loss_train, running_batch_loss_test, running_batch_accuracy_train, running_batch_accuracy_test



if __name__ == '__main__':

    # Initiate the transformer model
    transformer_model = TransformerEncoder(feat_dim=35,
                                           max_len=40,
                                           d_model=35, 
                                           n_heads=7, 
                                           num_layers=1,
                                           dim_feedforward=256, 
                                           dropout=0.1, 
                                           freeze=False)
    
    transformer_model.float()

    # Load pretrained weights
    transformer_model.load_state_dict(torch.load('../models/inputting_standard_norm.pt'))

    # Initiate the CNN model
    cnn_model = CNNModel().float()

    # Combine both models
    main_model = CombinedModel(transformer_model, cnn_model).float()

    # Load the dataset
    with open('../data/data_indices.json', 'r') as f: data_indices = json.load(f)
    train_indices = data_indices['train_indices']
    train_dataloader = DataLoader(MVTSDataset(train_indices, norm_type='standard', mean_mask_length=3, masking_ratio=0.15), batch_size=10, shuffle=True, drop_last=True)
    val_indices = data_indices['val_indices']
    test_dataloader = DataLoader(MVTSDataset(val_indices, norm_type='standard', mean_mask_length=3, masking_ratio=0.15), batch_size=10, shuffle=True, drop_last=True)

    # Train and validate the model
    n_epoch = 100
    train_loader = train_dataloader
    criterion = nn.CrossEntropyLoss()
    save_path = '../models/classification_standard_norm.pt'
    optimizer = torch.optim.Adam(main_model.parameters(), lr=0.001)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_model.to(device)
    train_loss, test_loss, train_acc, test_acc = train_and_validate_classifier(main_model,
                                                                               train_dataloader, 
                                                                               test_dataloader,
                                                                               n_epoch,
                                                                               save_path
    )