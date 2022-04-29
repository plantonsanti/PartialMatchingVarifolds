#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from IO                           import *
from registration_wrapper         import register_structure


common = './data/surfaces/'

"""
The paramters of the registration :
gamma          -> controls the regularity of the deformations
factor         -> if one changes the unit of the axes
sigmaV         -> the largest scale of the deformations 
sigmaW         -> the scales in the data attachment terms
max_iter_steps -> number of iteration per scale
method         -> the data attachment method, possible keywords:
                        - Varifold
                        - PartialVarifold
                        - PartialVarifoldLocal
                        - PartialVarifoldLocalNormalized 
"""

parameters = {
    "gamma": 10000,
    "sigmaV": 100,
    "sigmaW": [
        10,
        5
    ],
    "max_iter_steps": [
        100,
        500
    ],
    "method": "PartialVarifoldLocalNormalizedRegularized" # "Varifold" # 
}


folder2save = './results/surfaces/'
try_mkdir(folder2save)  

path_CBCT  = common+'/CBCT.npy'
path_CT    = common+'/CT.npy'

V_CBCT_np = np.load(path_CBCT, allow_pickle=True)[0]['points']
F_CBCT_np = np.load(path_CBCT, allow_pickle=True)[0]['cells'].astype(dtype = 'int32')             
#V_CBCT_np, F_CBCT_np =  RemoveDuplicates(V_CBCT_np,F_CBCT_np)

V_CT_np = np.load(path_CT, allow_pickle=True)[0]['points']
F_CT_np = np.load(path_CT, allow_pickle=True)[0]['cells'].astype(dtype = 'int32')
#V_CT_np, F_CT_np = RemoveDuplicates(V_CT_np,F_CT_np)

register_structure(V_CBCT_np, F_CBCT_np, V_CT_np, F_CT_np, folder2save, parameters=parameters,
                        structure = "Surfaces")



