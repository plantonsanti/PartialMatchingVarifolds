#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from IO                           import *
from registration_wrapper         import register_structure


common = './data/trees/'

parameters = {
    "Default": 0,
    "gamma": 10,
    "factor": 1,
    "sigmaV": 100,
    "sigmaW": [
        10,
        5
    ],
    "max_iter_steps": [
        200,
        200
    ],
    "method": "PartialVarifoldLocalNormalized" # "Varifold" # 
}

gamma,factor,sigmaV,sigmaW,max_iter_steps,method = read_parameters(parameters)


folder2save = './results/trees/'
try_mkdir(folder2save)  

path_template  = common+'/template.npz'
path_target    = common+'/target.npz'


V_template_np = np.load(path_template, allow_pickle=True)['points']
F_template_np = np.load(path_template, allow_pickle=True)['connections'].astype(dtype = 'int32')             

V_target_np = np.load(path_target, allow_pickle=True)['points']
F_target_np = np.load(path_target, allow_pickle=True)['connections'].astype(dtype = 'int32')


register_structure(V_template_np, F_template_np, 
                   V_target_np, F_target_np, 
                   folder2save, parameters=parameters, structure='Curves')


