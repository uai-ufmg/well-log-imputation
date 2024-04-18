# Dataset preprocessing for Imputation benchmark

This folder contains the script for preprocessing datasets for the imputation benchmark experiments and the arguments that it takes.

## Preprocessing Steps

This processing is divided in 6 steps as follows:
1. Load the dataset into a dataframe (either using a csv file, or reading .las files from a folder)
2. Selects the logs passed by the user
3. Clip outliers and normalize the selected logs
4. Perform a k-fold partition of the WELLS
5. For each well, crop slices from the entire log with a sliding window of length *n* and stride *s*
6. Save the croped slices into .npy files for future use.

## Basic use
```bash
# Process the geolink dataset found in `dataset_dir` using the wells wit las files located in `las_folder` using only three logs (GR, RHOB, NPHI)
python processing.py --dataset_name geolink --dataset_dir geolink_root --use_las --las_folder filtered_las --logs GR RHOB NPHI
``` 

### Key parameters:

| **Command**      | **Default**                   | **Description**                                                                                                                                  |
|------------------|-------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------|
| `--output_dir`   | 'preprocessed_data'           | Output folder where generated files are saved                                                                                                    |
| `--n_folds`      | 5                             | Number of folds that the dataset was partitioned                                                                                                 |
| `--slice_stride` | 256                           | The number of indices skipped from the starting point of last slice. Set equal or larger than the slice_size for no overlaping slices            |
| `--slice_size`   | 256                           | Length of the sliced sequences                                                                                                                   |
| `--seed`, `-s`   | 0                             | Random seed                                                                                                                                      |
| `--dataset_name` | 'geolink'                     | The name of the dataset (the name used to create the .npy files)                                                                                 |
| `--dataset_dir`  | 'geolink_root'           | Path to the processed dataset (the folder that contains the .npy files)                                                                          |
| `--use_las`      | False                         | If present, during load will try to read from .las files. By default tries to use a logs.csv file                                                |
| `--las_folder`   | 'filtered_las'                | The subfolder of `dataset_dir` where the .las files are located. Only used if `--use_las` is true                                                |
| `--logs`         | ['GR', 'DTC', 'RHOB', 'NPHI'] | List of the logs present in the processed dataset   (the order in the list is assumed as the same order of logs in  the stored processed slices) |