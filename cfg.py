'''
Config class: It is used to setup the paramenters to run the imputation benchmark
              During parsing, do some checks and generate the experiments list to be executed
'''

import os
from pathlib import Path
from argparse import ArgumentParser, Namespace
import random
import numpy as np
import torch

class Configs:
    def __init__(self):
        parser = ArgumentParser("Imputation Benchmarking")
        
        # general
        parser.add_argument("--output_dir", type=Path, default=Path("trained_models"), help="Path to save logs and model weights")
        parser.add_argument("--n_folds", type=int, default=5, help="Number of folds that the data is partioned")
        parser.add_argument("--seed", "-s", type=int, default=17076, help="Seed used for random generators")
        
        # dataset
        parser.add_argument("--dataset_name", type=str.lower, default="geolink", choices=["geolink", "taranaki", "teapot"])
        parser.add_argument("--dataset_dir", type=Path, default=Path("preprocessed_data"), help="Path to location of the preprocessed data (.npy files)")
        parser.add_argument("--logs", type=str.upper, nargs='+', default=['GR', 'DTC', 'RHOB', 'NPHI'],
                            help="named profiles presented on the data (in the order found in the arrays)")
        
        # experiment setup (Not used by shallow/classical methods)
        parser.add_argument("--slice_len", type=int, default=256, help="Length (number of samples) of the training/validation sequences")
        parser.add_argument("--epochs", type=int, default=1000, help="Total of training epochs.")
        parser.add_argument("--patience", type=int, default=1000, 
                            help="Number of patience epochs. Aborts training if validation score do not improve in this number of epochs.")
        parser.add_argument("--batch_size", type=int, default=32, help="Number of sequences in each training batch")
        
        # experiment missing pattern setup
        parser.add_argument("--missing_pattern", type=str.lower, nargs='+', default=['single', 'block', 'profile'],
                            help="missing patterns used to evaluate the trained models")
        parser.add_argument("--n_points", type=int, nargs='+', default=[1], help="number of random missing points in each sequence, defaults to 1")
        parser.add_argument("--blocks_size", type=int, nargs='+', default=[20, 100], help="length of the block of contigual missing points in each sequence")
        parser.add_argument("--profiles", type=str.upper, nargs='+', default=['RAND'])
        
        # model setup 
        parser.add_argument("--model", type=str.lower, default="saits", choices=['locf', 'mean', # classical (mean not implemented)
                                                                                 'rf', 'xgboost', 'svm', # shallow (encapsulations of sklearn implementation)
                                                                                 'saits', 'transformer', # attenttion time-series (pypots implementation)
                                                                                 'brits', 'mrnn', # rnn time-series (pypots implementation)
                                                                                 'unet', # cnn based method (monai backbone implementation)
                                                                                 'ae', # autoencoder with mlp (fully-connected) (our implementation)
                                                                                 ])
        parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate used in the optimizer")  # (Not used by shallow/classical methods)   
        parser.add_argument("--optimizer", type=str.lower, default='adam', choices=['adam', 'adamw'], 
                            help="Name of the optimizer used in training") # (Not used by shallow/classical methods) 
        
        
        self.parser = parser

        
    def parse_args(self):
        
        args = self.parser.parse_args()
        
        # Create output dir if does not exist
        if not args.output_dir.exists():
            Path.mkdir(args.output_dir, parents=True, exist_ok=True)
            
        # Check if dataset directory exists
        if not args.dataset_dir.exists():
            print(f"The provided dataset path `{args.dataset_dir}` does not exist. Aborting program execution.")
            exit()
        
        # setup torch device
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # number of features in the data
        args.n_features = len(args.logs)
        
        # generating experiment list
        args.experiments = []
        
        for mp in args.missing_pattern:
            if mp not in ['single', 'block', 'profile']:
                print(f"Missing pattern `{mp}` not implemented. Choose one (or more) from ['single', 'block', 'profile'].")
                exit()
                
            
            if mp == 'single':
                for n in args.n_points:
                    exp = {}
                    exp['mode'] = f'single.{n}'
                    exp['n_points'] = n
                    exp['b_size'] = None
                    exp['f_pos'] = None
                    
                    args.experiments.append(exp)
                    
            elif mp == 'block':
                for b in args.blocks_size:
                    exp = {}
                    exp['mode'] = f'block.{b}'
                    exp['n_points'] = None
                    exp['b_size'] = b
                    exp['f_pos'] = None
                    
                    args.experiments.append(exp)
                    
            elif mp == 'profile':
                for p in args.profiles:
                    if p != 'RAND' and p not in args.logs:
                        print(f"Profile named `{p}` not found in provided log list {args.logs}. Select a profile from the provided log list.")
                        exit()
                        
                    exp = {}
                    exp['mode'] = f'profile.{p}' if p != "RAND" else 'profile'
                    exp['n_points'] = None
                    exp['b_size'] = None
                    exp['f_pos'] = None if p == "RAND" else args.logs.index(p)
                    
                    args.experiments.append(exp)
                    
        
        return args