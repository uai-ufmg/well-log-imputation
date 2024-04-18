'''
This file contains the encapsulation of shallow ML methods to follow the design pattern of PyPOTS. (https://github.com/WenjieDu/PyPOTS/)
That is, in this capsule classes, they implement a fit and predict functions.

This file contains the capsules of the following methods: 
    Random Forest (RF) and SVM (from sklearn https://scikit-learn.org/stable/ ), 
    and XGBoost (xgboost package https://xgboost.readthedocs.io/en/stable/python/python_intro.html )
'''

from pypots.imputation.base import BaseImputer
from typing import Union, Optional
import torch
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

class RF(BaseImputer):
    '''
    Random Forest based imputation method. 
    
    @attr num_models: total number of random forest models. It is equal to the number of features present in the input data
    @attr models: dict containing each of the models. The i-th model is trained to predicted values of the i-th feature
    '''
    
    def __init__(self, num_models: int, device: Optional[Union[str, torch.device, list]] = None, **kwargs) -> None:
        '''
        Construct a Random Forest based imputation method.
        
        @param num_models: total number of random forest models. It is equal to the number of features present in the input data
        @param device: device used to train the model. NOT USED: the training is perfomed solely on cpu
        @param **kwargs: keyword arguments passed to the sklearn RandomForestRegressor constructor. 
                         It includes parameters for model configuration or training criteria 
                         (See: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html#sklearn.ensemble.RandomForestRegressor )
        '''
        
        super().__init__(device=device)
        
        self.num_models = num_models    
        self.models = {}

        for i in range(num_models):
            self.models[i] = RandomForestRegressor(**kwargs)
        
    def fit(self, train_set: dict[str, np.ndarray], val_set: Optional[dict[str, np.ndarray]] = None) -> None:
        '''
        Trains each of the RF models
        
        @param train_set: the training data, is a dict with the masked missing data and a indicating mask
        @param val_set: optional validation data. NOT USED
        '''
        
        X = np.copy(train_set["X"])
        mask = train_set["indicating_mask"]
        
        for i in range(self.num_models):
            y = X[:,:,i].reshape(-1)
            x = np.delete(X,i,axis=2).reshape(-1,self.num_models-1)
            
            self.models[i].fit(x,y)
    
    def predict(self, val_set: dict[str, np.ndarray] ) -> dict[str, np.ndarray]:
        '''
        Predicts the missing values of the val_set data
        
        @param val_set: the validation data, is a dict with the masked missing data and a indicating mask
        
        @return pred: a dict with the predicted values
        '''
        
        pred = {}
        
        X = np.copy(val_set["X"])
        mask = val_set["indicating_mask"]
        X = np.nan_to_num(X)
        X_pred = []
        # For each feature i, use the i-th model to predict its values 
        # and fill the missing spots with the predicted ones
        for i in range(self.num_models):
            y = X[:,:,i:i+1]
            
            y_pred = np.zeros(shape=(X.shape[0],X.shape[1],1))
            y_pred[mask[:,:,i] == 0] = y[mask[:,:,i] == 0]
            
            r = mask[:,:,i] == 0
            
            x = np.delete(X,i,axis=2)[mask[:,:,i] == 1]
            if x.shape[0] > 0:
                model_pred = self.models[i].predict(x).reshape(-1,1)
                y_pred[mask[:,:,i] == 1] = model_pred
            
            X_pred.append(y_pred)
            
        pred["imputation"] = np.concatenate(X_pred,axis = 2)
        
        return pred
        
    def impute(self,):
        pass
    
class XGBOOST(BaseImputer):
    def __init__(self, num_models: int, device: Optional[Union[str, torch.device, list]] = None, **kwargs) -> None:
        '''
        Construct a XGBoost based imputation method.
        
        @param num_models: total number of models. It is equal to the number of features present in the input data
        @param device: device used to train the model. NOT USED: the training is perfomed solely on cpu
        @param **kwargs: keyword arguments passed to the XGBRegressor constructor. 
                         It includes parameters for model configuration or training criteria 
                         (See: https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.XGBRegressor )
        '''
        
        super().__init__(device=device)
        
        self.num_models = num_models    
        self.models = {}

        for i in range(num_models):
            self.models[i] = XGBRegressor(**kwargs)
        
    def fit(self, train_set: dict[str, np.ndarray], val_set: Optional[dict[str, np.ndarray]] = None) -> None:
        '''
        Trains each of the XGBoost models
        
        @param train_set: the training data, is a dict with the masked missing data and a indicating mask
        @param val_set: optional validation data. NOT USED
        '''
        
        X = np.copy(train_set["X"])
        mask = train_set["indicating_mask"]
        
        for i in range(self.num_models):            
            y = X[:,:,i].reshape(-1)
            x = np.delete(X,i,axis=2).reshape(-1,self.num_models-1)
            
            self.models[i].fit(x,y)
    
    def predict(self, val_set: dict[str, np.ndarray] ) -> dict[str, np.ndarray]:
        '''
        Predicts the missing values of the val_set data
        
        @param val_set: the validation data, is a dict with the masked missing data and a indicating mask
        
        @return pred: a dict with the predicted values
        '''
        
        pred = {}
        
        X = np.copy(val_set["X"])
        mask = val_set["indicating_mask"]
        X = np.nan_to_num(X)
        X_pred = []
        # For each feature i, use the i-th model to predict its values 
        # and fill the missing spots with the predicted ones
        for i in range(self.num_models):
            y = X[:,:,i:i+1]
            
            y_pred = np.zeros(shape=(X.shape[0],X.shape[1],1))
            y_pred[mask[:,:,i] == 0] = y[mask[:,:,i] == 0]
            
            r = mask[:,:,i] == 0
            
            x = np.delete(X,i,axis=2)[mask[:,:,i] == 1]
            if x.shape[0] > 0:
                model_pred = self.models[i].predict(x).reshape(-1,1)
                y_pred[mask[:,:,i] == 1] = model_pred
            X_pred.append(y_pred)
            
        pred["imputation"] = np.concatenate(X_pred,axis = 2)
        
        return pred
        
    def impute(self,):
        pass

    
class SVM(BaseImputer):
    def __init__(self, num_models: int, device: Optional[Union[str, torch.device, list]] = None, **kwargs) -> None:
        '''
        Construct a SVM based imputation method.
        
        @param num_models: total number of models. It is equal to the number of features present in the input data
        @param device: device used to train the model. NOT USED: the training is perfomed solely on cpu
        @param **kwargs: keyword arguments passed to the sklearn SVR constructor. 
                         It includes parameters for model configuration or training criteria 
                         (See: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html#sklearn.svm.SVR)
        '''
        
        super().__init__(device=device)
        
        self.num_models = num_models    
        self.models = {}

        for i in range(num_models):
            self.models[i] = SVR(**kwargs)
        
    def fit(self, train_set: dict[str, np.ndarray], val_set: Optional[dict[str, np.ndarray]] = None) -> None:
        '''
        Trains each of the SVR models
        
        @param train_set: the training data, is a dict with the masked missing data and a indicating mask
        @param val_set: optional validation data. NOT USED
        '''
        
        X = np.copy(train_set["X"])
        mask = train_set["indicating_mask"]
        
        for i in range(self.num_models):            
            y = X[:,:,i].reshape(-1)
            x = np.delete(X,i,axis=2).reshape(-1,self.num_models-1)
            
            self.models[i].fit(x,y)
    
    def predict(self, val_set: dict[str, np.ndarray] ) -> dict[str, np.ndarray]:
        '''
        Predicts the missing values of the val_set data
        
        @param val_set: the validation data, is a dict with the masked missing data and a indicating mask
        
        @return pred: a dict with the predicted values
        '''
        
        pred = {}
        
        X = np.copy(val_set["X"])
        mask = val_set["indicating_mask"]
        X = np.nan_to_num(X)
        X_pred = []
        # For each feature i, use the i-th model to predict its values 
        # and fill the missing spots with the predicted ones
        for i in range(self.num_models):
            y = X[:,:,i:i+1]
            
            y_pred = np.zeros(shape=(X.shape[0],X.shape[1],1))
            y_pred[mask[:,:,i] == 0] = y[mask[:,:,i] == 0]
            
            r = mask[:,:,i] == 0
            
            x = np.delete(X,i,axis=2)[mask[:,:,i] == 1]
            if x.shape[0] > 0:
                model_pred = self.models[i].predict(x).reshape(-1,1)
                y_pred[mask[:,:,i] == 1] = model_pred
            X_pred.append(y_pred)
            
        pred["imputation"] = np.concatenate(X_pred,axis = 2)
        
        return pred
        
    def impute(self,):
        pass