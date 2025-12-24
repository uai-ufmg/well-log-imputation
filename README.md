# Imputation in Well Log Data: A Benchmark

> Repository with the code for conducting the experiments of the paper ** [Imputation in Well Log Data: A Benchmark](https://www.sciencedirect.com/science/article/pii/S0098300424002723) **

Imputation of well log data is a common task in the field. However, a quick review of the literature reveals a lack of padronization when evaluating methods for the problem.
The goal of the benchmark is to introduce a standard evaluation protocol to any imputation method for well log data. A small summary of the benchmark can be seen [here](#).

The code is mainly built using the [PyTorch](https://pytorch.org/) deep learning library and the [PyPOTS](https://pypots.com/) module.

### Requirements
The exact versions of PyTorch and PyPOTS used were:
```
pypots==0.1.4
torch==2.1.0a0+29c30b1
```
Installing them with pip should download their correct dependencies.
Newer versions of both libraries will probably also work.

## Repository content and organization

- The `main.py` file is the principal script and is used to evaluate a method in a given dataset
- The `cfg.py` file contains all the arguments that the `main.py` script accepts. A complete list of arguments is found [here](#key-parameters).
- `analise.ipynb` a jupyter notebook containing some visualization of the results, `format_metric.ipynb` a notebook to collect and format the results from training logs into a `.csv` file
- `data` directory contains modules related to data handling and manipulation: 
  * `data/datasets.py` contains classes for preparing data for some of the imputation methods
  * `data/missing.py` contains methods to generate pseudo masks of missing data
- `models` directory contains modules related to the implementation of some imputation methods (following the design pattern of PyPOTS)
- `preprocessing` directory contains methods related to the preprocessing of the datasets required to run the benchmark experiments. See the [README](/preprocessing) of the preprocessing script for its arguments.
- `utils` directory contains utility methods, such as some additional error metrics.

## Getting Started

> IMPORTANT: Before evaluating a method using well log data, you first need to preprocess the dataset, dividing it into folds and slices. See the `preprocessing` [README](/preprocessing) for details on how to preprocess the data.

### Dataset folder structure

After preprocessing your dataset, the fold structure should look as below:

``` bash
dataset_root
├── dataset_name_fold_0_well_log_metadata_train.json       # general metadata file for training partition of fold 0
├── dataset_name_fold_0_well_log_metadata_val.json         # general metadata file for validation partition of fold 0
├── dataset_name_fold_0_well_log_slices_meta_train.json    # per well metadata file for training partition of fold 0
├── dataset_name_fold_0_well_log_slices_meta_val.json      # per well metadata file for validation partition of fold 0
├── dataset_name_fold_0_well_log_sliced_train.npy          # sliced logs of the training partition of fold 0 in .npy format
├── dataset_name_fold_0_well_log_sliced_val.npy            # sliced logs of the validation partition of fold 0 in .npy format
...
├── dataset_name_fold_k_well_log_metadata_train.json       # general metadata file for training partition of fold k
├── dataset_name_fold_k_well_log_metadata_val.json         # general metadata file for validation partition of fold k
├── dataset_name_fold_k_well_log_slices_meta_train.json    # per well metadata file for training partition of fold k
├── dataset_name_fold_k_well_log_slices_meta_val.json      # per well metadata file for validation partition of fold k
├── dataset_name_fold_k_well_log_sliced_train.npy          # sliced logs of the training partition of fold k in .npy format
└── dataset_name_fold_k_well_log_sliced_val.npy            # sliced logs of the validation partition of fold k in .npy format
```
*Here `k` represents the total number of folds you selected to divide the dataset. In our benchmark, we used `k=5`.*

## Examples of Use

Assume you have preprocessed the Geolink dataset, using 5 folds, 4 logs (GR, DTC, RHOB, NPHI), and slices of length 256. 
To test some imputation methods on the default experiments of the Benchmark you could run in the terminal:
```bash
# Evaluates the SAITS model on the default experiments of the benchmark in the Geolink dataset
python main.py --dataset_name geolink --dataset_folder geolink_dataset --n_folds 5 --logs GR DTC RHOB NPHI --model saits
```

To test some imputation method on specific missing patterns you could run in the terminal:
```bash
# Evaluates the SAITS model in the Geolink dataset using the Block missing pattern of length 150
python main.py --dataset_name geolink --dataset_folder geolink_dataset --n_folds 5 --logs GR DTC RHOB NPHI --model saits --missing_pattern block --b_size 150
```

### Key Parameters:

| **Command**         |                    **Default** | **Description**                                                                                                                                                                             |
|---------------------|-------------------------------:|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `--output_dir`      |               'trained_models' | Output folder where generated files are saved                                                                                                                                               |
| `--n_folds`         |                              5 | Number of folds that the dataset was partitioned                                                                                                                                            |
| `--seed`            |                          17076 | Random seed                                                                                                                                                                                 |
| `--dataset_name`    |                      'geolink' | The name of the dataset (the name used to create the .npy files)                                                                                                                            |
| `--dataset_dir`     |            'preprocessed_data' | Path to the processed dataset (the folder that contains the .npy files)                                                                                                                     |
| `--logs`            |  ['GR', 'DTC', 'RHOB', 'NPHI'] | List of the logs present in the processed dataset <br /> (the order in the list is assumed as the same order of logs in  the stored processed slices)                                              |
| `--slice_len`       |                            256 | Length of the sliced sequences of the processed dataset                                                                                                                                     |
| `--epochs`          |                           1000 | Number of training epochs for deep methods                                                                                                                                                  |
| `--patience`        |                           1000 | Number of patience epochs. If the validation score does not improve in this number of epochs, training is aborted                                                                               |
| `--batch_size`      |                             32 | Size of training batch                                                                                                                                                                      |
| `--missing_pattern` | ['single', 'block', 'profile'] | List of missing patterns to be tested. <br />  - Single: independently selects random samples <br /> - Block: randomly select contiguous block of samples <br /> - Profile: mask an entire log of a sequence  |
| `--n_points`        |                            [1] | List of possible individual samples to be masked. Defaults to 1, larger integers imply multiple random indices independently selected per sequence to be masked                          |
| `--blocks_size`     |                      [20, 100] | List of possible block lengths to be masked. Each value creates a new experiment                                                                                                         |
| `--profiles`        |                       ['RAND'] | List of logs to be masked, if 'profile' is one of the tested missing patterns (when 'RAND', each test sequence has a random log masked)                                                    |
| `--model`           |                        'saits' | Name of the model to be evaluated <br /> (options: 'locf', 'rf', 'xgboost', 'svm', 'saits', 'transformer','brits', 'mrnn','unet', 'ae')                                                            |
| `--lr`              |                           1e-3 | Learning rate used in the optimizer                                                                                                                                                         |
| `--optimizer`       |                         'adam' | Name of the optimizer used in training (either 'adam' or 'adamw')                                                                                                                            |

### Available models
- 'locf': Last Observation Carry Forward
- 'rf': Random Forest [[source]](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor)
- 'xgboost': XGBoost Regressor [[source]](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor)
- 'svm': Support Vector Regressor [[source]](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR)
- 'saits': SAITS (Self-Attention Imputation Time Series) [[source]](https://github.com/WenjieDu/PyPOTS/tree/main/pypots/imputation/saits) [[paper]](https://arxiv.org/pdf/2202.08516)
- 'transformer': Transformer based Imputation for Time Series [[source]](https://github.com/WenjieDu/PyPOTS/tree/main/pypots/imputation/transformer) [[paper]](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- 'brits': BRITS (Bidirectional Recurrent Imputation for Time Series) [[source]](https://github.com/WenjieDu/PyPOTS/tree/main/pypots/imputation/brits) [[paper]](https://papers.nips.cc/paper_files/paper/2018/file/734e6bfcd358e25ac1db0a4241b95651-Paper.pdf)
- 'mrnn': mRNN [[source]](https://github.com/WenjieDu/PyPOTS/tree/main/pypots/imputation/mrnn) [[paper]](https://ieeexplore.ieee.org/ielaam/10/8694044/8485748-aam.pdf?tag=1)
- 'unet': UNet based imputation method [[paper]](https://arxiv.org/pdf/1505.04597.pdf)
- 'ae': Autoencoder (NN) based imputation method

## Datasets

The benchmark uses three public well log datasets:

- Geolink: The Geolink Dataset is another public dataset of wells in the Norwegian offshore. The data is provided by the company of the same name, [GEOLINK](https://www.geolink-s2.com/) and follows the NOLD 2.0 license. 
  This dataset contains a total of 223 wells. It also has lithology labels for the wells with a total of 36 lithology classes. [[download]](https://drive.google.com/drive/folders/1EgDN57LDuvlZAwr5-eHWB5CTJ7K9HpDP)
- Taranaki Basin: The Taranaki Basin Dataset is a curated set of wells and a convenient option for experimentation especially due to it is ease of accessibility and use. This collection, under the CDLA-Sharing-1.0 license, contains well logs extracted from the [New Zealand Petroleum & Minerals Online Exploration Database](https://geodata.nzpam.govt.nz/) and [Petlab](http://pet.gns.cri.nz/). There are a total of 407 wells, of which 289 are onshore and 118 are offshore exploration and production wells. [[download]](https://developer.ibm.com/exchanges/data/all/taranaki-basin-curated-well-logs/)
- Teapot Dome: The Teapot Dome dataset is provided by the Rocky Mountain Oilfield Testing Center (RMOTC) and the US Department of Energy.
  It contains different types of data related to the Teapot Dome oil field, such as 2D and 3D seismic data, well logs, and GIS data. The data is licensed under the Creative Commons 4.0 license.
  In total, the dataset has 1,179 wells with available logs. The number of available logs varies across wells. There are only 91 wells with the gamma ray, bulk density, and neutron porosity logs, while only three wells have the complete basic suite. [[direct download]](http://s3.amazonaws.com/open.source.geoscience/open_data/teapot/rmotc.tar)

The three public datasets can be downloaded already processed as it was used in the benchmark experiments [here](https://zenodo.org/records/10987946).

## Adding a novel imputation method

To include a new imputation method in this base three steps have to be performed.

#### First
Create an implementation file `mymodel.py` in the models directory, that will include everything you need to define your imputation method.
In addition to your method definitions, it is essential that the class of your method implement two functions:
1. `fit(...)` function: it receives a train_set (a dictionary of strings to numpy arrays) with:
 - the training sequences with artificial missing values added
 - the original training sequences
 - an indicating mask with values equal to 1 where the artificial missing data was added
 
 The `fit` function trains your model with the passed data
 
2. `predict(...)` function: it receives a test_set (similar to the train_set above) and returns a dictionary with the test_set with imputed values

See `models/shallow.py` or `models/autoencoder.py` for examples of implementations of these two functions.

#### Second
You have to implement and include an `instantiate_MYMODEL()` function in the `models/__init__.py` in addition to importing your model implementation file in that same file. In `models/__init__.py` you can find the exact parameters passed to your function with examples of implementation for the available methods. It is in this `instantiate_MYMODEL()` function that the hyperparameters of your method have to be set.

#### Lastly
In the `cfg.py`, you have to add your model as an option in argument `--model`

With these three steps, your method is ready to be tested using this code implementation and you should be able to execute in the terminal:
```bash
# Evaluates mymodel on the default experiments of the benchmark in the Geolink dataset
python main.py --dataset_name geolink --dataset_folder geolink_dataset --n_folds 5 --logs GR DTC RHOB NPHI --model mymodel
```
to evaluate your method

---
# Cite us:
```
@article{gama2025imputation,
 title = {Imputation in well log data: A benchmark for machine learning methods},
 journal = {Computers & Geosciences},
 volume = {196},
 pages = {105789},
 year = {2025},
 issn = {0098-3004},
 doi = {https://doi.org/10.1016/j.cageo.2024.105789},
 url = {https://www.sciencedirect.com/science/article/pii/S0098300424002723},
 author = {Pedro H.T. Gama and Jackson Faria and Jessica Sena and Francisco Neves and Vinícius R. Riffel and Lucas Perez and André Korenchendler and Matheus C.A. Sobreira and Alexei M.C. Machado},
 keywords = {Well log data, Imputation, Machine learning, Benchmark},
 abstract = {Well log data are an important source of information about the geological patterns along the wellbore but may present missing values due to sensor failure, wellbore irregularities or the cost of acquisition. As a consequence, incomplete log sequences may impact the performance of machine learning (ML) models for classification or prediction. Although several approaches for this problem have been proposed in the literature, the lack of consistent evaluation protocols hinders the comparison of different solutions. This paper aims at bridging this gap by proposing a robust benchmark for comparing imputation ML methods. It contributes to establish a standardized experimental protocol that could be used by the petroleum industry in the development of new methodologies for this purpose. It differs from previous works that have been based on different datasets and metrics that prevent an unbiased comparison of results. Eight imputation methods were investigated: Autoencoders (AE), Bidirectional Recurrent Neural Network for Time Series Imputation, Last Observation Carry Forward (LOCF), Random Forests, Self Attention for Imputation of Time Series (SAITS), Transformers, UNet, and XGBoost. The Geolink, Taranaki and Teapot datasets were used to contemplate data from different locations, from which sequences of measurements were deleted and further imputed by the selected ML methods. The Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, Pearson Correlation Coefficient and the Determination Coefficient were used for performance assessment in a set of 480 experiments. The results demonstrated that simple methods as the LOCF and the AE provided competitive imputation results, although the overall best model was SAITS. This reveals that self-attention models are a promising trend for imputation techniques. The choice for the LOCF, AE, SAITS, UNet, and XGBoost to compose the proposed benchmark was corroborated by subsequent statistical analyses, showing that it can be considered a compromise between simplicity, unbiasedness, variety and meaningfulness.}
}
```
