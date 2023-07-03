import torch
import pickle
import numpy as np
from models import CNNModel
from sklearn import metrics
import torch.nn.functional as F
from datasets import MVTSDataset
from torch.utils.data import DataLoader

# Metrics

# Function to calculate AUC
def roc_auc_score(y_true, y_score):
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

# Function to calculate accuracy
def accuracy(y_true, y_hat):
    y_pred = np.where(y_hat >= 0.5, 1, 0)  # Thresholding at 0.5 to convert into binary predictions
    total_samples = len(y_true)
    correct_predictions = np.sum(y_true == y_pred)
    accuracy_score = correct_predictions / total_samples
    return accuracy_score

# Function to calculate the Brier skill score
def calculate_brier_skill_score(y_true, y_hat):
    brier_score = ((y_true - y_hat) ** 2).mean()
    climatology_brier_score = ((y_true.mean() - y_true) ** 2).mean()
    brier_skill_score = 1 - (brier_score / climatology_brier_score)
    return brier_skill_score

# Function to calculate HSS (Cohen's kappa coefficient)
def hss(y_true, y_hat):
    y_pred = np.where(y_hat >= 0.5, 1, 0)  # Thresholding at 0.5 to convert into binary predictions
    total_samples = len(y_true)
    a = np.sum(y_true == 1)  # Number of true positive predictions
    b = np.sum(y_true == 0)  # Number of true negative predictions
    c = np.sum(y_pred == 1)  # Number of positive predictions
    d = np.sum(y_pred == 0)  # Number of negative predictions
    p0 = (a + b) / total_samples  # Overall agreement
    pe = ((a + c) * (a + b) + (b + d) * (c + d)) / (total_samples ** 2)  # Expected agreement by chance
    hss_score = (p0 - pe) / (1 - pe)  # HSS score
    return hss_score

# Function to calculate TSS (Hanssen & Kuiper's skill score or Peirce skill score)
def tss(y_true, y_hat):
    y_pred = np.where(y_hat >= 0.5, 1, 0)  # Thresholding at 0.5 to convert into binary predictions
    tp = np.sum((y_true == 1) & (y_pred == 1))  # True positives
    tn = np.sum((y_true == 0) & (y_pred == 0))  # True negatives
    fp = np.sum((y_true == 0) & (y_pred == 1))  # False positives
    fn = np.sum((y_true == 1) & (y_pred == 0))  # False negatives
    hit_rate = tp / (tp + fn)  # True Positive Rate (TPR) or Hit Rate
    false_alarm_rate = fp / (fp + tn)  # False Positive Rate (FPR) or False Alarm Rate
    tss_score = hit_rate - false_alarm_rate  # True Skill Score (TSS)
    return tss_score

# Calculate metrics for ensemble of models to get robust statistics of validation set

if __name__ == '__main__':

    # Initialize lists to store metrics
    running_tss_val = []
    running_auc_val = []
    running_hss_val = []
    running_bss_val = []
    running_acc_val = []

    # Collect filenames for all 50 models
    path_to_splits = '../kfold/splits/'
    file_names = np.arange(0, 50, 1)
    file_names = np.delete(file_names, np.argwhere(file_names == 27))
    file_names = np.delete(file_names, np.argwhere(file_names == 37))

    # Loop over all 50 different splits
    for file_name in file_names:

        print(file_name)

        # Load split indices
        path = path_to_splits + 'fold_' + str(file_name) + '.npz'
        fhand = np.load(path)
        val_indices = fhand['val_indices']

        # Create dataloaders
        val_dataloader = DataLoader(MVTSDataset(val_indices, norm_type='standard'), batch_size=len(val_indices), shuffle=False, drop_last=False)

        # Load validation data
        data_val, _, labels_val = next(iter(val_dataloader))
        data_val = torch.nan_to_num(data_val)
        data_val = data_val.unsqueeze_(1)

        # Load model
        model = CNNModel()
        path = '../kfold/models/cnn_model_standard_' + str(file_name) + '.pth'
        model.load_state_dict(torch.load(path))
        model.eval()

        # Calculate y_hat and y_true for the validation set
        logits_val = model(data_val)
        probabilities_val = F.softmax(logits_val, dim=1) 
        y_hat_val = probabilities_val[:, 1].detach().numpy() 
        y_true_val = labels_val.detach().numpy()

        # Apply metrics
        tss_val = tss(y_true_val, y_hat_val)
        auc_val = roc_auc_score(y_true_val, y_hat_val)
        hss_val = hss(y_true_val, y_hat_val)
        bss_val = calculate_brier_skill_score(y_true_val, y_hat_val)
        accuracy_val = accuracy(y_true_val, y_hat_val)

        # Append metrics to lists
        running_tss_val.append(tss_val)
        running_auc_val.append(auc_val)
        running_hss_val.append(hss_val)
        running_bss_val.append(bss_val)
        running_acc_val.append(accuracy_val)
    


    # Store metics in a dictionary and save to disk
    metrics = {
        'running_tss_val': running_tss_val,
        'running_auc_val': running_auc_val,
        'running_hss_val': running_hss_val,
        'running_bss_val': running_bss_val,
        'running_acc_val': running_acc_val
    }
    with open('../data/metrics_standard_cnn.pkl', 'wb') as f: pickle.dump(metrics, f)