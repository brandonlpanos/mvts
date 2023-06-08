import torch
import numpy as np
from torch.utils.data import Dataset


class ImputationDataset(Dataset):
    """Dynamically computes missingness (noise) mask for each sample"""

    def __init__(self, data, indices, mean_mask_length=3, masking_ratio=0.15, mode='separate', distribution='geometric', exclude_feats=None):
        super(ImputationDataset, self).__init__()
        self.data = data  # this is a subclass of the BaseData class in data.py
        self.IDs = indices  # list of data IDs, but also mapping between integer index and ID
        self.feature_df = self.data.feature_df.loc[self.IDs]
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length
        self.mode = mode
        self.distribution = distribution
        self.exclude_feats = exclude_feats

    def update(self):
        self.mean_mask_length = min(20, self.mean_mask_length + 1)
        self.masking_ratio = min(1, self.masking_ratio + 0.05)

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, ind):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            X: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            mask: (seq_length, feat_dim) boolean tensor: 0s mask and predict, 1s: unaffected input
            ID: ID of sample
        """
        X = self.feature_df.loc[self.IDs[ind]].values  # (seq_length, feat_dim) array
        mask = noise_mask(X, self.masking_ratio, self.mean_mask_length, self.mode, self.distribution, self.exclude_feats)  # (seq_length, feat_dim) boolean array
        return torch.from_numpy(X), torch.from_numpy(mask), self.IDs[ind]
    

def noise_mask(X, masking_ratio, lm=3):
    """
    Creates a random boolean mask of the same shape as X, with 0s at places where a feature should be masked.
    Args:
        X: (seq_length, feat_dim) numpy array of features corresponding to a single sample
        masking_ratio: proportion of seq_length to be masked. At each time step, will also be the proportion of
            feat_dim that will be masked on average
        lm: average length of masking subsequences (streaks of 0s). Used only when `distribution` is 'geometric'.
    Returns:
        boolean numpy array with the same shape as X, with 0s at places where a feature should be masked
    """
    mask = np.ones(X.shape, dtype=bool)
    for m in range(X.shape[1]):  # feature dimension
        mask[:, m] = geom_noise_mask_single(X.shape[0], lm, masking_ratio)  # time dimension
    return mask

def geom_noise_mask_single(L, lm, masking_ratio):
    """
    Randomly create a boolean mask of length L, consisting of subsequences of average length lm, masking with 0s a masking_ratio
    proportion of the sequence L. The length of masking subsequences and intervals follow a geometric distribution.
    Args:
        L: length of mask and sequence to be masked
        lm: average length of masking subsequences (streaks of 0s)
        masking_ratio: proportion of L to be masked
    Returns:
        (L,) boolean numpy array intended to mask ('drop') with 0s a sequence of length L
    """
    keep_mask = np.ones(L, dtype=bool)
    p_m = 1 / lm  # probability of each masking sequence stopping. parameter of geometric distribution.
    p_u = p_m * masking_ratio / (1 - masking_ratio)  # probability of each unmasked sequence stopping. parameter of geometric distribution.
    p = [p_m, p_u]
    # Start in state 0 with masking_ratio probability
    state = int(np.random.rand() > masking_ratio)  # state 0 means masking, 1 means not masking
    for i in range(L):
        keep_mask[i] = state  # here it happens that state and masking value corresponding to state are identical
        if np.random.rand() < p[state]:
            state = 1 - state
    return keep_mask

    
