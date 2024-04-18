'''
Missing data related file
This file contains methods to create random masks for sequences (or batch of sequences) to artificially simulate missing values
'''

from typing import Optional
import numpy as np

def mask_single(interval: np.ndarray, n_points: Optional[int] = None, feat: Optional[int] = None) -> np.ndarray:
    '''
    Randomly selects `n_points` indices of the `interval` to be masked.
    If `feat` is None, a random feature of the interval is selected, otherwise the mask is applied to the `feat` column
    
    @param interval: a np.ndarray containing the sequence. Shape (sequece_len, num_features)
    @param n_points: an optional value indicating the number of random indices to be selected. If None, a single point is selected
    @param feat: an optional value indicating the feature column that the mask is applied. If None a random column is selected
    
    @return mask: a bool np.ndarray with True values where the data should be considered missing. Shape (sequece_len, num_features)
    '''
    if feat is None:
        feat = np.random.randint(0, interval.shape[1])
        
    if n_points is None:
        idx = np.random.randint(0, interval.shape[0])
    else:
        idx = np.random.randint(0, interval.shape[0], size=n_points)
    
    mask = np.zeros(interval.shape, dtype=bool)
    mask[idx, feat] = 1
    
    return mask

def mask_block(interval: np.ndarray, block_size: int = 20, feat: Optional[int] = None) -> np.ndarray:
    '''
    Randomly selects a contiguous block of `block_size` indices of the `interval` to be masked.
    If `feat` is None, a random feature of the interval is selected, otherwise the mask is applied to the `feat` column
    
    @param interval: a np.ndarray containing the sequence. Shape (sequece_len, num_features)
    @param block_size: a value indicating the length of the block of indices to be selected. 
    @param feat: an optional value indicating the feature column that the mask is applied. If None a random column is selected
    
    @return mask: a bool np.ndarray with True values where the data should be considered missing. Shape (sequece_len, num_features)
    '''
    
    if feat is None:
        feat = np.random.randint(0, interval.shape[1])
        
    init = np.random.randint(0, interval.shape[0] - block_size)
    
    mask = np.zeros(interval.shape, dtype=bool)
    mask[init:init+block_size, feat] = 1
    
    return mask

def mask_profile(interval: np.ndarray, feat: Optional[int] = None) -> np.ndarray:
    '''
    Selects a column of the `interval` to be masked.
    If `feat` is None, a random feature of the interval is selected, otherwise `feat` column is masked
    
    @param interval: a np.ndarray containing the sequence. Shape (sequece_len, num_features)
    @param feat: an optional value indicating the feature column that the mask is applied. If None a random column is selected
    
    @return mask: a bool np.ndarray with True values where the data should be considered missing. Shape (sequece_len, num_features)
    '''
    if feat is None:
        feat = np.random.randint(0, interval.shape[1])
    
    mask = np.zeros(interval.shape, dtype=bool)
    mask[:, feat] = 1
    
    return mask

def mask_X(X: np.ndarray, mode: str = 'single', 
           n_points: int = 1, 
           b_size: list[int] = [20, 100], 
           feat: Optional[int] = None, 
           nan: float = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    '''
    Applies a desired (`mode`) missing pattern to all sequences of the data X
    
    @param X: the data, an np.ndarray with all the sequences. Shape (num_sequences, sequence_len, num_features)
    @param mode: a str indicating the missing pattern to be applied. It can be one of the following values
                    - `single` uses the mask_single function above
                    - `block` uses the mask_block function above
                    - `profile` uses the mask_profile function above
                    - `rand` for each sequence, randomly apply one of the patterns above 
    @param n_points: the value passed to the mask_single(interval, n_points). That is, the number of randomly selected indices per sequence to be masked
    @param b_size: a list with values to be passed to the mask_block(interval, block_size). That is, the length of contiguous indices selected per sequence to be masked.
                   If the list has more than one item, the value is randomly selected;
    @param feat: the value passed to the mask_profile(interval, feat). That is, the column (feature) of the sequence to be masked
    @param nan: the value used to fill artificial missing data at the sequences 
    
    @return a tuple with: the original data, 
                          the data with artificial missing values,
                          the missing mask (=0 where the data was masked, 1 otherwise) and,
                          the indicating mask (=1 where the data was masked, 0 otherwise)
    '''
    
    assert mode in ['single', 'block', 'profile', 'rand']
    
    _mode = mode
    
    X = np.copy(X)
    X_intact = np.copy(X)  # keep a copy of originally observed values in X_intact
    
    # get masks for each interval in X
    masks = []
    for interval in X:
        if mode == 'rand': _mode = np.random.choice(['single', 'block', 'profile'])
        
        if _mode == 'single':
            mask = mask_single(interval, n_points=n_points, feat=feat)
        
        if _mode == 'block':
            mask = mask_block(interval, block_size=np.random.choice(b_size), feat=feat)
            
        if _mode == 'profile':
            mask = mask_profile(interval, feat=feat)
        
        masks.append(mask)
    
    masks = np.array(masks)
    
    X[masks] = np.nan  # mask values selected by missing_mask
    
    indicating_mask = ((~np.isnan(X_intact)) ^ (~np.isnan(X))).astype(np.float32)
    missing_mask = (~np.isnan(X)).astype(np.float32)                               # 0 where X is nan (missing), 1 otherwise
    X_intact = np.nan_to_num(X_intact, nan=nan)
    X = np.nan_to_num(X, nan=nan)
    
    return tuple((X_intact, X, missing_mask, indicating_mask))
    