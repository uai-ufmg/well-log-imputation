"""
Evaluation metrics related to error calculation (like in tasks regression, imputation etc).
"""

from typing import Union, Optional
import numpy as np
import torch
from sklearn import metrics
from scipy.spatial import distance

def cal_r2(class_predictions: Union[np.ndarray, torch.Tensor],
           targets: Union[np.ndarray, torch.Tensor],
           masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
          ) -> Union[float, torch.Tensor]:
    '''
    Calculate the R-squared Error between ``class_predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    @param class_predictions: The prediction data to be evaluated.
    @param targets: The target data for helping evaluate the predictions.
    @param masks: The masks for filtering the specific values in inputs and target from evaluation.
                  When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.
    '''

    R2 = metrics.r2_score(targets.flatten(), class_predictions.flatten(), sample_weight=masks.flatten())

    return R2

def cal_cc(class_predictions: Union[np.ndarray, torch.Tensor],
           targets: Union[np.ndarray, torch.Tensor],
           masks: Optional[Union[np.ndarray, torch.Tensor]] = None,
          ) -> Union[float, torch.Tensor]:
    '''
    Calculate the Correlation Distance (=1 - Correlation) between ``class_predictions`` and ``targets``.
    ``masks`` can be used for filtering. For values==0 in ``masks``,
    values at their corresponding positions in ``predictions`` will be ignored.

    @param class_predictions: The prediction data to be evaluated.
    @param targets: The target data for helping evaluate the predictions.
    @param masks: The masks for filtering the specific values in inputs and target from evaluation.
                  When given, only values at corresponding positions where values ==1 in ``masks`` will be used for evaluation.
    '''

    CC = distance.correlation(targets.flatten(), class_predictions.flatten(), w=masks.flatten())

    return CC