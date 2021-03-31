#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import os
import torch
import numpy as np

from keops_utils import TestCuda

use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod = TestCuda(verbose=False)


def try_mkdir(path):
    """
    return 0 if folder already exists, 1 if created.
    """
    r = 0

    try:
        os.mkdir(path)
        r = 1
    except OSError:
        pass

    return r
    

default_parameters = {  "default" : True,
                        "gamma" : 0.1,
                        "factor" : 1,
                        "sigmaV" : 100,
                        "sigmaW" : [100,25],
                        "max_iter_steps" : [100,500],
                        "method" : "ConstantNormalCycle",
                        "template" : False
                     }


def read_parameters(parameters = {}):
    """
    Read the parameters in a dictionary. If they are not provided, the default 
    parameters are taken.
    
    @param : parameters        : dictionary : supposed to contain : 
                                    - default : boolean: if True, set the paramters to default. 
                                    - method  : string : data attachment method ("ConstantNormalCycle", "LinearNormalCycle", 
                                                                                "CombineNormalCycle", "PartialVarifold",
                                                                                "Varifold")
                                    - factor  : float  : the factor scale to homogenize the losses if the scale changes. 
                                    - sigmaV  : float  : scale of the diffeomorphism.
                                    - sigmaW  : list of float : the different scales of the data attachment term. 
                                    - max_iter_steps : list of integers : must be same size as sigmaW, the number of iteration 
                                                                          per data attachment scale. 
                                    - template : boolean : if we want to build a template from the registration to all the targets. 
                                    
    """

    if("gamma" in parameters.keys()):
        gamma = parameters["gamma"]
    else:
        gamma = torch.tensor([default_parameters["gamma"]], dtype=torchdtype, device=torchdeviceId)
    if("factor" in parameters.keys()):
        factor = parameters["factor"]
    else:
        factor = default_parameters["factor"]
    if("sigmaV" in parameters.keys()):
        scaleV = factor*parameters["sigmaV"]
    else:
        scaleV = factor*default_parameters["sigmaV"]
    sigmaV = torch.tensor([scaleV], dtype=torchdtype, device=torchdeviceId)
    
    if("sigmaW" in parameters.keys()):
        sigmaW = [factor*torch.tensor([sigW], dtype=torchdtype, device=torchdeviceId) for sigW in parameters["sigmaW"]]
    else:
        sigmaW = [factor*torch.tensor([sigW], dtype=torchdtype, device=torchdeviceId) for sigW in default_parameters["sigmaW"]]
    
    if("max_iter_steps" in parameters.keys()):
        max_iter_steps = parameters["max_iter_steps"]
    else:
        max_iter_steps = default_parameters["max_iter_steps"]
    if(len(max_iter_steps)!=len(sigmaW)):
        max_iter_steps = [50 for s in sigmaW]
        
    if("method" in parameters.keys()):
        method = parameters["method"]
    else:
        method = default_parameters["method"]
    
    if("template" in parameters.keys()):
        use_template=parameters["template"]
    else:
        use_template=default_parameters["template"]

    return gamma,factor,sigmaV,sigmaW,max_iter_steps,method,use_template



def RemoveDuplicates(Points,Connections):
    """

    """

    p_, indices, counts = np.unique(Points, axis=0, return_index=True, return_counts=True)

    order = indices.argsort()
    ind_ordered = np.sort(indices)
    counts_ordered = counts[order]
    p_clean = Points[ind_ordered,:]

    n_con = Connections.shape[0]
    mask_keep = np.ones(n_con,dtype=np.int8)
   
    #mask_keep = np.ones_like(Connections)

    for i,ind in enumerate(ind_ordered):
        if(counts_ordered[i]>1): #then this point is several times in the mesh
            print('New ind : ', ind, 'order : ', counts_ordered[i])
            p_ref = Points[ind]
            cpt = 0
            for k,p in enumerate(Points):
                if (p==p_ref).all() and  k!=ind: #then it is a dupe, and not the first
                   
                    print("Duplicate : ", k)
                    for ind_con in range(n_con):
                        #To remove all the connections to a dupe
                        if k-cpt in Connections[ind_con,:]:
                            mask_keep[ind_con]=0
                           
                        #To update the connections values
                        if Connections[ind_con,0]>k-cpt :
                            Connections[ind_con,0] = Connections[ind_con,0]-1
                        if Connections[ind_con,1]>k-cpt :
                            Connections[ind_con,1] = Connections[ind_con,1]-1
                        if Connections[ind_con,2]>k-cpt :
                            Connections[ind_con,2] = Connections[ind_con,2]-1
                    cpt+=1
                   
    c_clean = Connections[mask_keep==1,:] #np.ma.masked_array(Connections, mask_keep).data

    return p_clean, c_clean
    
    
    
