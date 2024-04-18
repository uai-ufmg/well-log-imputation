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

def filter_by_logs_petro(intervals, well_logs, log_filter):
    log_mask = []
    for i, log in enumerate(well_logs):
        if log in log_filter:
            log_mask.append(i)

    well_logs = well_logs[log_mask]
    intervals = [x[:, log_mask] for x in intervals]
    intervals = np.array(intervals, dtype=object)


    return intervals, well_logs
    
def load_data_petro (input_path, selected_well_logs =[], verbose=False):
    '''
    Load well log data at input_path
    
    @param input_path path to directory containing the well log data.
    @param selected_well_logs optional parameter, if not empty only well_logs passed in this list will be returned
    @param verbose for debugging.
    
    @return tuple with 4 elements: np.array with the intervals, list with the name of each interval, list with the name of the wells, name of the retrieved well logs
    '''
    
    intervals = np.load(os.path.join(input_path, 'np_intervals.npy'), allow_pickle=True)
    well_logs = np.load(os.path.join(input_path, 'np_well_log_names.npy'))

    if len(selected_well_logs) == 0:
        selected_well_logs = well_logs

    if verbose:
        print (f'Conjunto originais de perfis: {well_logs}')
        print (f'Conjunto filtrado de perfis: {selected_well_logs}')
    
    intervals, selected_well_logs  = filter_by_logs_petro(intervals, well_logs, selected_well_logs)

    
    f = os.path.join(input_path,'intervals_labels.json')
    well_names = np.array ( json.load(open(f), object_pairs_hook=lambda x : [ str(k).split('_')[0] for _, k in x ]  ) )
    interval_names = np.array ( json.load(open(f), object_pairs_hook=lambda x : [ str(k) for _, k in x ]  ) )
    
    if verbose == True:
        print ()
        for interval, label in zip(intervals, well_names):
            print (label, interval.shape)
        print ()

    return intervals, interval_names, well_names, selected_well_logs

def load_petro(dataset_dir, selected_well_logs=[]):
    intervals, interval_names, well_names, selected_well_logs = load_data_petro(dataset_dir, selected_well_logs=selected_well_logs)
    
    concat_df = None
    for id, interval in enumerate(intervals):
        df = pd.DataFrame(interval, columns=selected_well_logs)
        df = df.assign(INTERVAL = interval_names[id])

        if concat_df is None:
            concat_df = df
        else:
            concat_df = pd.concat([concat_df, df])
            
    return concat_df


def load_data(dataset_dir, selected_logs, std_names, is_petro, use_las, las_folder=None, rname_logs = []):
    data = None

    if is_petro:
        ################################
        # dataframe from internal data #
        ################################
        data = load_petro(dataset_dir, rname_logs)

    else:
        if use_las:
            ##################################
            # dataframe from las file folder #
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
            # dataframe from csv file #
            ###########################

            with open( os.path.join(dataset_dir, 'logs.csv'), 'rb' ) as f:
                data = pd.read_csv(f, index_col=0)
    
    data = data.rename(std_names, axis='columns')
    return data

def winsorization(data, logs):
    for c in logs:
        array_data = data[c].values

        ## Calculate percentiles for the limits
        min_, max_ = np.nanquantile(array_data, q=[0.01, 0.99])
        print(f'{c}: min: {min_:.4f} - max: {max_:.4f} ')

        ## Set outlier values as nan:
        outlier_idx = np.logical_or(array_data < min_, array_data > max_)
        print(f'ignoring {np.sum(outlier_idx)} values')
        array_data[outlier_idx] = np.nan

        ## Set series in dataframe with clipped values
        data[c] = array_data
        # data = data.assign(**{c: array_data})

        print()
    return data

def normalize_data(data, logs):
    scaler = StandardScaler()
    data[logs] = scaler.fit_transform(data[logs])
    
    return data

def get_valid_intervals(log_array):
    log_is_notnan = ~ np.isnan(log_array)
    all_valid = np.all(log_is_notnan, axis=1)
    breaks = [0] + (np.where(all_valid[:-1] != all_valid[1:])[0] + 1).tolist() + [all_valid.size]

    valid_intervals = []
    for i in range(len(breaks) - 1):
        ## Store the starting and ending index of the interval, if it has observed values
        if all_valid[ breaks[i] ]:
            valid_intervals.append( (breaks[i], breaks[i + 1]) )
    
    return valid_intervals

def slice_wells(data, slice_size=256, slice_stride=256, selected_logs=['GR', 'DTC', 'RHOB', 'NPHI'], verbose=False):

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
            # print(len(valid_intervals), valid_intervals[:min(10, len(valid_intervals)-1)])
            # continue
        for interval in valid_intervals:
            start = interval[0]
            interval_max_depth = interval[1] - interval[0]
            # print(interval, interval_max_depth)
            
            while (start - interval[0]) < interval_max_depth:
                end = start + slice_size
                
                ## Check if slice overflow interval length:
                if (end - interval[0]) > interval_max_depth:
                    break
        
                # slice_data = well_data.iloc[start:end, :]
                # slice_data_array = slice_data[selected_logs].values
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


def process_data(data, dataset_name, n_folds, output_dir, slice_size=256, slice_stride=256, selected_logs=['GR', 'DTC', 'RHOB', 'NPHI'], seed=14273):
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
    # load data
    data = load_data(args.dataset_dir, args.logs, args.std_names, args.is_petro, args.use_las, args.dataset_las_folder, args.rname_logs)
    
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