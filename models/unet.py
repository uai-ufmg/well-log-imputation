'''
UNet implementation file. 
The UNet model used is from the monai library (https://docs.monai.io/en/stable/networks.html#unet)
This implementation follows the design of implementation of other imputation models of the PyPots module: https://github.com/WenjieDu/PyPOTS/
See an example of implementation: https://github.com/WenjieDu/PyPOTS/blob/main/pypots/imputation/saits/model.py
'''

from typing import Iterable, Union, Optional
import torch
import os
import numpy as np
import json

from sklearn.preprocessing import StandardScaler
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

from monai.networks.nets import UNet as mUNet
from torch.utils.data import DataLoader

from data import DatasetForImputation
from .neuralnet import _reconstruction_loss

from pypots.imputation.base import BaseNNImputer
from pypots.data.base import BaseDataset
from pypots.utils.logging import logger
from pypots.utils.metrics import cal_mae
from pypots.optim import Adam, AdamW


class UNet(BaseNNImputer):
    
    def __init__(self, n_features: int, 
                 spatial_dims: int = 1,
                 channels: Iterable[int] = tuple([2**(i+5) for i in range(5)]), 
                 strides: Iterable[int] = (2,2,2,2),
                 num_res_units: int = 1,
                 batch_size: int = 32,
                 epochs: int = 100,
                 patience: Optional[int] = None,
                 optimizer: Adam | AdamW = AdamW(lr=0.001),
                 num_workers: int = 0,
                 device: Optional[Union[str, torch.device, list]] = None,
                 saving_path: Optional[str] = None,
                 model_saving_strategy: str = "best"):
        '''
        Construct an imputation model based on an UNet architecture. 
        It is constructed upon the BaseNNImputer implement by pypots: https://github.com/WenjieDu/PyPOTS/blob/main/pypots/imputation/base.py#L134 
        
        @param n_features: the number of features in each sequence
        @param spatial_dims: the number of dimensions of the data (1d, 2d, 3d, in the case of log sequences the dim = 1)
        @param channels: the number of channels (feature maps) in each covolutional block
        @param strides: the stride of the convolutional filters in each block
        @param num_res_units: number of residual units
        @param batch_size: the size of the training batch,
        @param epochs: the number of training epochs
        @param patience: the number of patience epochs
        @param optimizer: the optimizer object used to train the model
        @param num_workers: number of workers (parallel process for training)
        @param device: training device (cpu or cuda usually)
        @param saving_path: path where model weights and logs are saved
        @param model_saving_strategy: strategy to save models 
        '''
        
        super().__init__(
            batch_size,
            epochs,
            patience,
            num_workers,
            device,
            saving_path,
            model_saving_strategy,
        )
    
        self.n_features = n_features,

        self.model_params =  { 'spatial_dims' : spatial_dims, 
                               'in_channels' : n_features,
                               'out_channels' : n_features,
                               'channels'  : channels, 
                               'strides' : strides, 
                               'num_res_units' : num_res_units,
                             }
        
        # Construct the monai model with the passed parameters
        self.model = mUNet(**self.model_params)

        self._print_model_size()
        self._send_model_to_given_device()

        self.optimizer = optimizer
        self.optimizer.init_optimizer(self.model.parameters())
    
    def _assemble_input_for_training(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        '''
        Prepate the input data and send it to the training device
        '''
        
        indices, X, X_intact, missing_mask, indicating_mask = self._send_data_to_given_device(data)

        inputs = {
            "X": X,
            "X_intact": X_intact,
            "missing_mask": missing_mask,         # 0 where data is nan, 1 otherwise
            "indicating_mask": indicating_mask,   # 1 where data is nan, 0 otherwise
        }

        return inputs

    def _assemble_input_for_validating(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._assemble_input_for_training(data)

    def _assemble_input_for_testing(self, data: torch.Tensor) -> dict[str, torch.Tensor]:
        return self._assemble_input_for_validating(data)

    def _train_model(self, training_loader: DataLoader, val_loader: DataLoader) -> None:
        '''
        Training process for the unet model
        
        @param training_loader: pytorch dataloader with the training data
        @param val_loader: pytorch dataloader with the validation data. Can be None in this case no validation metrics are computated
        '''
        
        # each training starts from the very beginning, so reset the loss and model dict here
        self.best_loss = float("inf")
        self.best_model_dict = None

        try:
            training_step = 0
            for epoch in range(self.epochs):
                self.model.train()
                epoch_train_loss_collector = []
                # training loop
                # the unet is trained to reconstruct the intact input using the faulty input X
                # (the ideia is to learn the correlations between features and be able to predict if one or more is missing, 
                #  this is done by giving emphasis on the reconstruction of missing values)
                for idx, data in enumerate(training_loader):
                    training_step += 1
                    
                    self.optimizer.zero_grad()
                    
                    inputs = self._assemble_input_for_training(data)
                    results = {}
                    
                    results["imputed_data"] = self.model.forward(inputs["X"])
                    
                    # It gives a larger weight for the missing data than to the non-masked data
                    l_rec = _reconstruction_loss (results["imputed_data"], inputs["X"], inputs["missing_mask"] )
                    l_imp =  5 * _reconstruction_loss (results["imputed_data"], inputs["X_intact"], inputs["indicating_mask"] )
                    results["loss"] = l_rec + l_imp
                    
                    results["loss"].backward()
                    self.optimizer.step()
                    epoch_train_loss_collector.append(results["loss"].item())

                    # save training loss logs into the tensorboard file for every step if in need
                    if self.summary_writer is not None:
                        self._save_log_into_tb_file(training_step, "training", results)

                # mean training loss of the current epoch
                mean_train_loss = np.mean(epoch_train_loss_collector)

                if val_loader is not None:
                    self.model.eval()
                    imputation_collector = []
                    with torch.no_grad():
                        for idx, data in enumerate(val_loader):
                            inputs = self._assemble_input_for_validating(data)
                            results = {}
                            
                            results["imputed_data"] = self.model.forward(inputs["X"])
                            imputation_collector.append(results["imputed_data"])

                    imputation_collector = torch.cat(imputation_collector)
                    imputation_collector = imputation_collector.moveaxis(1, -1).cpu().detach().numpy()

                    mean_val_loss = cal_mae(
                        imputation_collector,
                        val_loader.dataset.X_intact.numpy(),
                        val_loader.dataset.indicating_mask.numpy(),
                    )

                    # save validating loss logs into the tensorboard file for every epoch if in need
                    if self.summary_writer is not None:
                        val_loss_dict = {
                            "imputation_loss": mean_val_loss,
                        }
                        self._save_log_into_tb_file(epoch, "validating", val_loss_dict)

                    logger.info(
                        f"epoch {epoch}: "
                        f"training loss {mean_train_loss:.4f}, "
                        f"validating loss {mean_val_loss:.4f}"
                    )
                    mean_loss = mean_val_loss
                else:
                    logger.info(f"epoch {epoch}: training loss {mean_train_loss:.4f}")
                    mean_loss = mean_train_loss

                if mean_loss < self.best_loss:
                    self.best_loss = mean_loss
                    self.best_model_dict = self.model.state_dict()
                    self.patience = self.original_patience
                    # save the model if necessary
                    self._auto_save_model_if_necessary(
                        training_finished=False,
                        saving_name=f"{self.__class__.__name__}_epoch{epoch}_loss{mean_loss}",
                    )
                else:
                    self.patience -= 1

                if os.getenv("enable_nni", False):
                    nni.report_intermediate_result(mean_loss)
                    if epoch == self.epochs - 1 or self.patience == 0:
                        nni.report_final_result(self.best_loss)

                if self.patience == 0:
                    logger.info(
                        "Exceeded the training patience. Terminating the training procedure..."
                    )
                    break

        except Exception as e:
            logger.error(f"Exception: {e}")
            if self.best_model_dict is None:
                raise RuntimeError(
                    "Training got interrupted. Model was not trained. Please investigate the error printed above."
                )
            else:
                RuntimeWarning(
                    "Training got interrupted. Please investigate the error printed above.\n"
                    "Model got trained and will load the best checkpoint so far for testing.\n"
                    "If you don't want it, please try fit() again."
                )

        if np.equal(self.best_loss.item(), float("inf")):
            raise ValueError("Something is wrong. best_loss is Nan after training.")

        logger.info("Finished training.")
        
        
    def fit(self, 
            train_set: dict[str, np.ndarray | torch.Tensor], 
            val_set: dict[str, np.ndarray | torch.Tensor] | None = None) -> None:
        '''
        Fit process. It encapsulates the entire training procedure. First it prepares the data, then train the model and save the weights if necessary
        
        @param train_set: the training data is a dict {'X': masked data,
                                                       'X_intact': original data,
                                                       'indicating_mask': indicating mask (=1 where the data was corrupted)}
        @param val_set: the validation data can be a dict {'X': masked data,
                                                           'X_intact': original data,
                                                           'indicating_mask': indicating mask (=1 where the data was corrupted)} 
                        or None (in this case, no validation metrics are computed)
        '''
        
        # Step 1: wrap the input data with classes Dataset and DataLoader
        training_set = DatasetForImputation(train_set)
        training_loader = DataLoader(training_set,
                                     batch_size=self.batch_size,
                                     shuffle=True,
                                     num_workers=self.num_workers,)

        val_loader = None
        if val_set is not None:
            val_set = DatasetForImputation(val_set)
            val_loader = DataLoader(val_set,
                                    batch_size=self.batch_size,
                                    shuffle=False,
                                    num_workers=self.num_workers,)


        # Step 2: train the model and freeze it
        self._train_model(training_loader, val_loader)
        self.model.load_state_dict(self.best_model_dict)
        self.model.eval()  # set the model as eval status to freeze it.

        # Step 3: save the model if necessary
        self._auto_save_model_if_necessary(training_finished=True)


    def predict(self, test_set: dict[str, np.ndarray | torch.Tensor], file_type: str = "h5py") -> dict[str, np.ndarray]:
        '''
        Predict process. It encapsulates a prediction procedure. 
        First it prepares the data, process it with a trained model and returns the predicted values (including the missing ones) 
        
        @param test_set: the test data a dict {'X': masked data,
                                               'X_intact': original data,
                                               'indicating_mask': indicating mask (=1 where the data was corrupted)}
        @param file_type: the format in with the test data is saved in memory. NOT USED
        
        @return result_dict: a dictionary with the predicted values (np.ndarray)
        '''
        
        # Step 1: wrap the input data with classes Dataset and DataLoader

        self.model.eval()  # set the model as eval status to freeze it.
        test_set = DatasetForImputation(test_set)
        test_loader = DataLoader(test_set,
                                 batch_size=self.batch_size,
                                 shuffle=False,
                                 num_workers=self.num_workers,)

        imputation_collector = []

        # Step 2: process the data with the model
        with torch.no_grad():
            for idx, data in enumerate(test_loader):
                inputs = self._assemble_input_for_testing(data)
                
                results = self.model.forward(inputs["X"])
                imputation_collector.append(results)

        # Step 3: output collection and return
        imputation = torch.cat(imputation_collector).moveaxis(1, -1).cpu().detach().numpy()
        result_dict = {
            "imputation": imputation,
        }

        return result_dict
    
    def impute(self, X: np.ndarray | torch.Tensor, file_type: str = "h5py") -> torch.Tensor:
        '''
        Deprecated predict method. See above
        '''
        
        logger.warning("🚨DeprecationWarning: The method impute is deprecated. Please use `predict` instead.")

        results_dict = self.predict(X, file_type=file_type)
        return results_dict["imputation"]
