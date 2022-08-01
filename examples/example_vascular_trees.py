#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys 
import os
sys.path.append(os.path.abspath("../utils"))
sys.path.append(os.path.abspath("../registration"))

import numpy as np

from IO                           import *
from registration_wrapper         import register_structure


"""
The paramters of the registration :
gamma          -> controls the regularity of the deformations
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
    "Default": 0,
    "gamma": 1,
    "factor": 1,
    "sigmaV": 50,
    "sigmaW": [
        50,
        25,
        5
    ],
    "max_iter_steps": [
        100,
        100,
        200
    ],
    "method": "Varifold" # "PartialVarifoldLocalNormalized" # 
}

common = '../data/trees/'
folder2save = '../results/vascular_trees/'
try_mkdir(folder2save)  

path_template  = common+'/template.npz'
path_target    = common+'/target.npz'


V_template_np = np.load(path_template, allow_pickle=True)['points']
F_template_np = np.load(path_template, allow_pickle=True)['connections'].astype(dtype = 'int32')             

V_target_np = np.load(path_target, allow_pickle=True)['points']
F_target_np = np.load(path_target, allow_pickle=True)['connections'].astype(dtype = 'int32')


register_structure(V_template_np, F_template_np, 
                   V_target_np, F_target_np, 
                   folder2save, parameters=parameters, structure='curves', reg_root = True)


