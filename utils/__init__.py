'''
Utility module.
Contains implementations of some regression metrics in a pattern similar to the Pypots: https://github.com/WenjieDu/PyPOTS/blob/main/pypots/utils/metrics/error.py
'''

from .metrics import cal_r2, cal_cc

__all__ = ["cal_r2", 
           "cal_cc"]