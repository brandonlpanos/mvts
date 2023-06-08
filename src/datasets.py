import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class ImputationDataset(Dataset):
    """Dynamically computes missingness (noise) mask for each sample"""

    def __init__(self, indicies, norm_type='unity', mean_mask_length=3, masking_ratio=0.15):
        super(ImputationDataset, self).__init__()

        self.indicies = indicies
        self.norm_type = norm_type

        if self.norm_type == 'standard':
            dataframes = []
            for file_name in os.listdir('../data/long/'):
                df = pd.read_csv('../data/long/' + file_name)
                dataframes.append(df)
            df = pd.concat(dataframes, ignore_index=True)
            df = df.drop(['Unnamed: 0'], axis=1)
            df = df.drop("R_VALUE", axis=1)
            df = df.drop("target", axis=1)
            df = df.to_numpy()
            # Apply standardization
            scaler = StandardScaler()
            scaler.fit(df)

            self.scaler = scaler
        self.masking_ratio = masking_ratio
        self.mean_mask_length = mean_mask_length

    def __len__(self):
        return len(self.indicies)

    def __getitem__(self, idx):
        """
        For a given integer index, returns the corresponding (seq_length, feat_dim) array and a noise mask of same shape
        Args:
            ind: integer index of sample in dataset
        Returns:
            x: (seq_length, feat_dim) tensor of the multivariate time series corresponding to a sample
            mask: (seq_length, feat_dim) boolean tensor: 0s mask and predict, 1s: unaffected input
        """
        index = self.indices[idx]
        df = pd.read_csv(f'../data/long/{index}.csv')
        if len(df) < 40:
            padding = pd.DataFrame(np.nan, index=np.arange(40 - len(df)), columns=df.columns)
            df = pd.concat([padding, df])

        label = df['target'].values[-1].astype(np.int64)
        df = df.drop("target", axis=1)
        df = df.drop("Unnamed: 0", axis=1)
        df = df.drop("R_VALUE", axis=1)
        x = np.array(df.values, dtype=np.float32)

        if self.norm_type == 'standard':
            x = self.scaler.transform(x)
        elif self.norm_type == 'unity':
            x = x.T
            x = self.unity_based_normalization(x)
            x = x.T

        mask = noise_mask(x, self.masking_ratio, self.mean_mask_length)  # (seq_length, feat_dim) boolean array

        return torch.from_numpy(X), torch.from_numpy(mask), label
    

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

    




def find_pad(mvts: torch.Tensor) -> torch.Tensor:
    """
    Takes in a batch of data shaped (batch_size, seq_length, feat_dim) and 
    returns a mask shaped (batch_size, seq_length) where 1 == True == Keep, 0 == False == Mask
    """
    # Create mask
    # (batch_size, seq_length) We say 1 == True == Keep, 0 == False == Mask, but Pytorch oposite, so within the code we flip the mask
    mask = torch.full(mvts.shape[0:-1], 1, dtype=torch.bool)
    mask[torch.all(mvts == 0, dim=-1)] = 0
    return mask

def padding_mask(lengths, max_len=None):
    """
    Used to mask padded positions: creates a (batch_size, max_len) boolean mask from a tensor of sequence lengths,
    where 1 means keep element at this position (time step)
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max_val()  # trick works because of overloading of 'or' operator for non-boolean types
    return (torch.arange(0, max_len, device=lengths.device)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))

