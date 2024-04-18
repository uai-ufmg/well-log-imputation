import os
from pathlib import Path
from argparse import ArgumentParser, Namespace
import random
import numpy as np
import torch


class Arguments:
    def __init__(self):
        parser = ArgumentParser("Imputation Preprocessing")
        
        # general
        parser.add_argument("--debug", "-d", action="store_true", default=False) # Not implemented
        parser.add_argument("--output_dir", type=Path, default="/pgeoprj/godeep/ej0p/preprocessed_data")
        parser.add_argument("--n_folds", type=int, default=5)
        parser.add_argument("--slice_stride", type=int, default=256)
        parser.add_argument("--slice_size", type=int, default=256)
        parser.add_argument("--seed", "-s", type=int, default=0)
        
        # dataset
        parser.add_argument("--dataset_name", type=str.lower, default="geolink", choices=["geolink", "taranaki", "teapot", "petro"])
        parser.add_argument("--dataset_dir", type=str, default="/pgeoprj/godeep/dados/l2_datasets/publico/geolink")
        parser.add_argument("--use_las", action="store_true", default=False, help="If during load the dataset uses las files or a csv")
        parser.add_argument("--las_folder", type=str, default="filtered_las")
        parser.add_argument("--logs", type=str.upper, nargs='+', default=['GR', 'DTC', 'RHOB', 'NPHI'],
                            help="named profiles extracted from the log files")
        
        self.parser = parser
        
    def parse_args(self):
        
        args = self.parser.parse_args()
        
        # Create output dir if does not exist
        if not args.output_dir.exists():
            Path.mkdir(args.output_dir, parents=True)
        
        if args.dataset_name == "geolink":
            std_names = {'DEN': 'RHOB', 'DENS': 'RHOB',
                         'NPOR': 'NPHI', 'PHIN': 'NPHI', 
                         'WELLNAME': 'WELL', 'WELL_NAME': 'WELL',
                        }
            args.rname_logs = []
            args.is_petro = False
        elif args.dataset_name == "taranaki":
            std_names = {'DEN': 'RHOB', 'DENS': 'RHOB',
                         'NPOR': 'NPHI', 'PHIN': 'NPHI', 
                         'WELLNAME': 'WELL', 'WELL_NAME': 'WELL',
                        }
            args.rname_logs = []
            args.is_petro = False
        elif args.dataset_name == "teapot":
                std_names = {'DEN': 'RHOB', 'DENS': 'RHOB',
                             'NPOR': 'NPHI', 'PHIN': 'NPHI', 
                             'WELLNAME': 'WELL', 'WELL_NAME': 'WELL',
                            }
                args.rname_logs = []
                args.is_petro = False
        elif args.dataset_name == "petro":
                std_names = {'DENSIDADE': 'RHOB', 'DT_CISALHANTE': 'DTS',
                             'DT_COMPRESSIONAL': 'DTC', 'MACRO_RESIST_LONGA': 'MRL', 
                             'INTERVAL': 'WELL',  'FATOR_FOTOELETRICO': 'PEF',
                            }
                args.reverse_names = {'RHOB': 'DENSIDADE', 'DTS': 'DT_CISALHANTE',
                                      'DTC': 'DT_COMPRESSIONAL', 'MRL': 'MACRO_RESIST_LONGA', 
                                      'PEF': 'FATOR_FOTOELETRICO',
                                     }
                args.rname_logs = [args.reverse_names[l] if l in args.reverse_names.keys() else l for l in args.logs]
                args.is_petro = True
        else:
            std_names = {}
            args.rname_logs = []
            args.is_petro = False
            
        args.std_names = std_names
        
        if args.use_las:
            args.dataset_las_folder = os.path.join(args.dataset_dir, args.las_folder)
        else:
            args.dataset_las_folder = ''
        
        return args