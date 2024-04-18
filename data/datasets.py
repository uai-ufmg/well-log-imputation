'''
Dataset implementation files
This file contains implementations of Datasets (derived from torch datasets) that prepare the data to be used for the imputation
models during training/validation
'''

import numpy as np
import torch
from torch.utils.data import Dataset

class DatasetForImputation(Dataset):
    '''
    Dataset class for our imputation methods implemented.
    
    @attr X: a float torch.Tensor containing the data with artificially missing data introduced. Shape (n_samples, seq_len, n_feat)
    @attr X_intact: a float torch.Tensor containing the original data. Shape (n_samples, seq_len, n_feat)
    @attr indicating_mask: a bool torch.Tensor indicating where the artificial missing data were introduced, =1 where the data is missing. Shape (n_samples, seq_len, n_feat)
    @attr y: an optional int torch.Tensor containing the label of the data. Shape (n_samples, seq_len)
    '''
    
    def __init__(self, data: dict[str, np.ndarray | torch.Tensor]) -> None:
        """
        Construct the dataset object
        
        @param data: is a dict {'X': masked data,
                                'X_intact': original data,
                                'indicating_mask': indicating mask (=1 where the data was corrupted)}
        """
        
        super().__init__()
        self.X = data['X']                                               # [n_samples, seq_len, n_feat] float
        self.X_intact = data['X_intact']                                 # [n_samples, seq_len, n_feat] float
        self.indicating_mask = data['indicating_mask']                   # [n_samples, seq_len, n_feat] bool
        self.y = None if "y" not in data.keys() else data["y"]           # [n_samples, seq_len]         int: optional labels
        
        # Convert data to tensor
        if isinstance(self.X, np.ndarray): self.X = torch.from_numpy(self.X)
        if isinstance(self.X_intact, np.ndarray): self.X_intact = torch.from_numpy(self.X_intact)
        if isinstance(self.indicating_mask, np.ndarray): self.indicating_mask = torch.from_numpy(self.indicating_mask)
        if isinstance(self.y, np.ndarray): self.y = torch.from_numpy(self.y)
        
    def __len__(self) -> int:
        '''The length (total number of sequences) of the dataset'''
        return len(self.X)
    
    def __getitem__(self, idx) -> list[torch.Tensor]:
        '''Gets the single sample at index `idx` of the dataset'''
        
        # its is transposing the data, because in pytorch the features are usually in the first dimension 
        X = self.X[idx].T.to(torch.float32)
        X_intact = self.X_intact[idx].T.to(torch.float32)
        indicating_mask = self.indicating_mask[idx].T.to(torch.float32)
        missing_mask = self.indicating_mask[idx].T.to(torch.bool)
        missing_mask = ~missing_mask
        missing_mask = missing_mask.to(torch.float32)
        X = torch.nan_to_num(X)
        sample = [
            torch.tensor(idx),
            X,
            X_intact,
            missing_mask,
            indicating_mask,
        ]

        if self.y is not None:
            sample.append(self.y[idx].to(torch.long))

        return sample