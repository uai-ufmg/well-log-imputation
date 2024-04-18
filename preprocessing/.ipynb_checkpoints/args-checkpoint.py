'''
Arguments class: It is used to setup the paramenters to run the preprocessing procedure
                 During parsing, do some checks and setup some parameters specific for each dataset
'''

import os
from pathlib import Path
from argparse import ArgumentParser, Namespace

class Arguments:
    def __init__(self):
        parser = ArgumentParser("Imputation Preprocessing")
        
        # general
        parser.add_argument("--output_dir", type=Path, default="preprocessed_data", help="path to the folder were the generate processed .npy will be stored")
        parser.add_argument("--n_folds", type=int, default=5, help="number of partitions (folds) that the wells of the dataset will be split")
        parser.add_argument("--slice_stride", type=int, default=256, 
                            help="the number of indices skiped from the starting point of last slice. Set equal or larger than the slice_size for no overlaping slices")
        parser.add_argument("--slice_size", type=int, default=256, help="number of indices (samples) in each slice of a log")
        parser.add_argument("--seed", "-s", type=int, default=0, help="random seed")
        
        # dataset
        parser.add_argument("--dataset_name", type=str.lower, default="geolink", choices=["geolink", "taranaki", "teapot"])
        parser.add_argument("--dataset_dir", type=str, default="geolink_root",
                           help="location of the dataset")
        parser.add_argument("--use_las", action="store_true", default=False, help="If during the dataset load uses las files or a csv")
        parser.add_argument("--las_folder", type=str, default="filtered_las", help="path of the las files of the dataset. Is assumed the folder is a subfolder of `dataset_dir`")
        parser.add_argument("--logs", type=str.upper, nargs='+', default=['GR', 'DTC', 'RHOB', 'NPHI'],
                            help="named profiles extracted from the log files, and the order that they will be stored")
        
        self.parser = parser
        
    def parse_args(self):
        
        args = self.parser.parse_args()
        
        # Create output dir if does not exist
        if not args.output_dir.exists():
            Path.mkdir(args.output_dir, parents=True)
        
        # setup a standart names for the profiles
        if args.dataset_name == "geolink":
            std_names = {'DEN': 'RHOB', 'DENS': 'RHOB',
                         'NPOR': 'NPHI', 'PHIN': 'NPHI', 
                         'WELLNAME': 'WELL', 'WELL_NAME': 'WELL',
                        }
        elif args.dataset_name == "taranaki":
            std_names = {'DEN': 'RHOB', 'DENS': 'RHOB',
                         'NPOR': 'NPHI', 'PHIN': 'NPHI', 
                         'WELLNAME': 'WELL', 'WELL_NAME': 'WELL',
                        }
        elif args.dataset_name == "teapot":
                std_names = {'DEN': 'RHOB', 'DENS': 'RHOB',
                             'NPOR': 'NPHI', 'PHIN': 'NPHI', 
                             'WELLNAME': 'WELL', 'WELL_NAME': 'WELL',
                            }
        else:
            std_names = {}
            
        args.std_names = std_names
        
        # setup the complete path of the las folder is necessary
        if args.use_las:
            args.dataset_las_folder = os.path.join(args.dataset_dir, args.las_folder)
        else:
            args.dataset_las_folder = ''
        
        return args