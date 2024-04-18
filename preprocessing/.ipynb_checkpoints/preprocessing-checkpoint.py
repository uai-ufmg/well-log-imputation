'''
Preprocessing file. Includes functions to load well log datasets, standardize, and processs for use in the imputation benchmark
'''

from typing import Optional
import missingno as msno
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import time
import json
import lasio

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold

from itertools import product
from tqdm import tqdm


def load_data(dataset_dir: str, selected_logs: list[str], 
              std_names: dict[str, str],
              use_las: bool, las_folder: Optional[str] = None) -> pd.DataFrame:
    '''
    Loads a well log dataset into a pandas DataFrame object.
    
    @param dataset_dir: the complete path where the dataset is located
    @param selected_logs: list with the name of the logs to be loaded. Ex: [GR, NPHI, RHOB]
    @param std_names: a dict with the names of the logs and the mapping to the standardized names
    @param use_las: a bool indicating if the loaded data will be from .las files or .csv
    @param las_folder: in case of loading from las files, this indicates the path to the folder containing said files
    
    @return a pandas DataFrame with the well log data
    '''
    
    data = None

    if use_las:
        ##################################
        # DataFrame from las file folder #
        ##################################

        for las in os.listdir(las_folder):
            if '.las' not in las and '.LAS' not in las: continue
            las_file = os.path.join(las_folder, las)
            las_obj = lasio.read(las_file)
            df = las_obj.df().reset_index()
            df = df.rename(args.std_names, axis='columns')
            y = df.columns.to_list()

            if 'GR' not in y:
                if 'GRN' in y:
                    df = df.rename({'GRN': 'GR'}, axis='columns')
                elif 'GRD' in y:
                    df = df.rename({'GRD': 'GR'}, axis='columns')


            df1 = df[selected_logs]
            df1 = df1.assign(WELL = las.split('/')[-1].split('.')[0])

            if data is None:
                data = df1
            else:
                data = pd.concat([data, df1], ignore_index=True)

    else:
        ###########################
        # DataFrame from csv file #
        ###########################

        with open( os.path.join(dataset_dir, 'logs.csv'), 'rb' ) as f:
            data = pd.read_csv(f, index_col=0)
    
    data = data.rename(std_names, axis='columns')
    return data

def winsorization(data: pd.DataFrame, logs: list[str]) -> pd.DataFrame:
    '''
    Applies the winsorization to the `data`. 
    It values smaller than the 1% percentil and larger than the 99% percentil are considered as outliers and tranformed into NaN
    
    @param data: the DataFrame with the values to be clipped
    @param logs: the logs to which the procedure will be applied
    
    @return a DataFrame with the values of `logs` clipped
    '''
    
    for c in logs:
        array_data = data[c].values

        ## Calculate percentiles for the limits
        min_, max_ = np.nanquantile(array_data, q=[0.01, 0.99])
        print(f'{c}: min: {min_:.4f} - max: {max_:.4f} ')

        ## Set outlier values as nan:
        outlier_idx = np.logical_or(array_data < min_, array_data > max_)
        print(f'ignoring {np.sum(outlier_idx)} values')
        array_data[outlier_idx] = np.nan

        ## Set series in DataFrame with clipped values
        data[c] = array_data
        # data = data.assign(**{c: array_data})

        print()
    return data

def normalize_data(data: pd.DataFrame, logs: list[str]) -> pd.DataFrame:
    '''
    Applies normalization to the `data`. 
    Transforms the data to follow a Normal distribution with mean 0 and std 1
    
    @param data: the DataFrame with the values to be normalized
    @param logs: the logs to which the procedure will be applied
    
    @return a DataFrame with the values of `logs` normalized
    '''
    
    scaler = StandardScaler()
    data[logs] = scaler.fit_transform(data[logs])
    
    return data

def get_valid_intervals(log_array: np.ndarray) -> list[tuple[int, int]]:
    '''
    Get all the valid intervals of array, Valid intervals are the ones without NaN values
    
    @param log_array: np.ndarray containing the data. shape (num_samples, num_logs)
    
    @return a list with the starting and ending point of all valid intervals in the original array
    '''
    
    log_is_notnan = ~ np.isnan(log_array)
    all_valid = np.all(log_is_notnan, axis=1)
    breaks = [0] + (np.where(all_valid[:-1] != all_valid[1:])[0] + 1).tolist() + [all_valid.size]

    valid_intervals = []
    for i in range(len(breaks) - 1):
        ## Store the starting and ending index of the interval, if it has observed values
        if all_valid[ breaks[i] ]:
            valid_intervals.append( (breaks[i], breaks[i + 1]) )
    
    return valid_intervals

def slice_wells(data: pd.DataFrame, 
                slice_size: int = 256, 
                slice_stride: int = 256, 
                selected_logs: list[str] = ['GR', 'DTC', 'RHOB', 'NPHI'], 
                verbose: bool = False) -> tuple[np.ndarray, list, list]:
    '''
    Slices the well logs in `data` into intervals of length `slice_size` with spacing between the starting point of adjacent intervals of `slice_stride`.
    Only logs in the `selected_logs` list are included in the sliced intervals
    
    @param data: pd.DataFrame containing the well log data. 
                 Assumes that data has a column named WELL with a identifier to each well, as well as columns presented in `selected_logs`
    @param slice_size: the length of the sliced intervals
    @param slice_stride: the spacing between starting points of adjacent intervals. For no overlaping this values needs to be >= `slice_size`
    @param selected_logs: list with the desired logs to be included in the sliced intervals. Assumes the names passed are present in the `data` DataFrame
    @param verbose: for debugging
    
    @return 3 tuple with: a np.ndarray containing the sliced intervals, 
                          a list of metadata per well (well_id, starting and ending points of the intervals), and 
                          a list of general metadata (well_ids processed, number of slices per well)
    '''
    
    slices_list = []
    slices_metadata = []
    fold_metadata = []
    
    well_ids = data.WELL.dropna().unique()

    for well_id in well_ids:
        sz = len(slices_list)
        well_data = data.loc[data.WELL == well_id, selected_logs].values
        well_max_depth = well_data.shape[0]
        
        ## Get the array with the log from the well data
        log_array = data.loc[data.WELL == well_id, selected_logs].values

        ## Get valid intervals intervals
        valid_intervals = get_valid_intervals(log_array)
        log_array = None
        
        if verbose:
            print(f'Processing well `{well_id}`')
            
        for interval in valid_intervals:
            start = interval[0]
            interval_max_depth = interval[1] - interval[0]
            
            while (start - interval[0]) < interval_max_depth:
                end = start + slice_size
                
                ## Check if slice overflow interval length:
                if (end - interval[0]) > interval_max_depth:
                    break
        
                slice_data_array = well_data[start:end, :]

                slices_list.append(slice_data_array)

                ## Save slice metadata
                slices_metadata.append((
                    well_id,                                         ## Well ID
                    start,                                           ## Starting index for the slice (inside well_data DataFrame)
                    min(end, well_max_depth),                        ## Final index for the slice (inside well_data DataFrame)
                    -1,
                    -1,
                    -1,
                    -1
                    # slice_data.DEPTH_MD.min(),                       ## Starting depth in well
                    # slice_data.DEPTH_MD.max(),                       ## Final depth in well
                    # check_value( slice_data.X_LOC.mean() ),          ## Average slice X position
                    # check_value( slice_data.Y_LOC.mean() )           ## Average slice Y position
                ))

                start = start + slice_stride
        
        if verbose:
            print(f'Valid slices for well `{well_id}`: {len(slices_list) - sz}')
        
        fold_metadata.append((well_id, len(slices_list) - sz))
    
    fold_metadata = [('_all', len(slices_list))] + fold_metadata
            
    return np.stack(slices_list), slices_metadata, fold_metadata


def process_data(data: pd.DataFrame,
                 dataset_name: str, 
                 n_folds: int, 
                 output_dir: str,
                 slice_size: int = 256,
                 slice_stride: int = 256,
                 selected_logs: list[str] = ['GR', 'DTC', 'RHOB', 'NPHI'],
                 seed: int = 14273) -> None:
    '''
    Process a dataset for the use in the imputation benchmark. It divides the wells into `n_folds` groups for training/validation
    For each fold, computes slices for all well in that fold training and validation partitions. These slices are saved into .npy files for future use in adition to some metadata
    The well logs in `data` are sliced into intervals of length `slice_size` with spacing between the starting point of adjacent intervals of `slice_stride`.
    Only logs in the `selected_logs` list are included in the sliced intervals
    
    @param data: pd.DataFrame containing the well log data. 
                 Assumes that data has a column named WELL with a identifier to each well, as well as columns presented in `selected_logs`
    @param dataset_name: a string with the name of the dataset being processed. Used for naming the generated files
    @param n_folds: the number of partitions to divide the wells
    @param outuput_dir: the path to the folder where the generated files will be saved
    @param slice_size: the length of the sliced intervals
    @param slice_stride: the spacing between starting points of adjacent intervals. For no overlaping this values needs to be >= `slice_size`
    @param selected_logs: list with the desired logs to be included in the sliced intervals. Assumes the names passed are present in the `data` DataFrame
    @param seed: the random seed used to create the folds
    '''
    
    well_ids = data.WELL.dropna().unique()

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for i, (train_index, test_index) in enumerate(kf.split(well_ids)):
        train_wells = sorted( np.array(well_ids)[train_index].tolist() )
        val_wells   = sorted( np.array(well_ids)[test_index].tolist() )

        print(f"Fold {i}:")
        print(f"{len(train_index), len(test_index), len(train_index)/len(well_ids), len(test_index)/len(well_ids)}")
        print(f"  Train wells: {well_ids[train_index]}")
        print(f"  Val wells: {well_ids[test_index]}")

        # separate train/val data
        train_data = data.loc[data.WELL.map(lambda x: x in train_wells)]
        val_data   = data.loc[data.WELL.map(lambda x: x in val_wells)]

        print("-- Getting slices ...")
        # get valid slices from training wells
        train_array_data, train_slices_metadata, train_metadata = slice_wells(train_data, 
                                                                              slice_size=slice_size, 
                                                                              slice_stride=slice_stride, 
                                                                              selected_logs=selected_logs)

        # get valid slices from validation wells
        val_array_data, val_slices_metadata, val_metadata = slice_wells(val_data, 
                                                                        slice_size=slice_size, 
                                                                        slice_stride=slice_stride, 
                                                                        selected_logs=selected_logs)

        print(train_array_data.shape, val_array_data.shape)

        print("-- Saving files ...")
        ## Save training data
        with open( os.path.join(output_dir, f'{dataset_name.lower()}_fold_{i}_well_log_sliced_train.npy'), 'wb' ) as f:
            np.save(file=f, arr=train_array_data)

        ## Save validation data
        with open( os.path.join(output_dir, f'{dataset_name.lower()}_fold_{i}_well_log_sliced_val.npy'), 'wb' ) as f:
            np.save(file=f, arr=val_array_data)

        ## Save training dataset metadata
        with open( os.path.join(output_dir, f'{dataset_name.lower()}_fold_{i}_well_log_slices_meta_train.json'), 'w' ) as f:
            json.dump(train_slices_metadata, f)
            
        with open( os.path.join(output_dir, f'{dataset_name.lower()}_fold_{i}_well_log_metadata_train.json'), 'w' ) as f:
            json.dump(train_metadata, f)

        ## Save validation dataset metadata
        with open( os.path.join(output_dir, f'{dataset_name.lower()}_fold_{i}_well_log_slices_meta_val.json'), 'w' ) as f:
            json.dump(val_slices_metadata, f)
            
        with open( os.path.join(output_dir, f'{dataset_name.lower()}_fold_{i}_well_log_metadata_val.json'), 'w' ) as f:
            json.dump(val_metadata, f)
        

def main(args):
    '''
    This script can be used to load and process a dataset. 
    For a comprehensive list of arguments see the Readme.md or the args.py file
    
    Example use:
    python processing.py --dataset_name geolink --dataset_dir geolink_root --use_las --las_folder filtered_las --logs GR,RHOB,NPHI
    '''
    # load data
    data = load_data(args.dataset_dir, args.logs, args.std_names, args.use_las, args.dataset_las_folder)
    
    # normalize data
    data = winsorization(data, args.logs)
    data = normalize_data(data, args.logs)
    
    process_data(data, args.dataset_name, args.n_folds, args.output_dir, 
                 slice_size=args.slice_size, slice_stride=args.slice_stride, 
                 selected_logs=args.logs, seed=args.seed)

if __name__ == '__main__':
    from args import Arguments
    args = Arguments().parse_args()
    print(args)
    main(args)