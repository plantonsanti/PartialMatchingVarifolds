#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import time
import optimization
import numpy as np
import json

from IO                           import *
from registration                 import *
from keops_utils                  import *

from data_attachment_surfaces import SurfacesDataloss
from data_attachment_curves   import CurvesDataloss

# Cuda management
use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod = TestCuda()


def register_structure(template, template_connections,
                        target, target_connections,
                        folder2save,
                        parameters={"default" : True},
                        structure = "Surfaces"):
    """
    Perform the registration of a source onto a target with an initialization as barycenter registration.
    
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

    gamma,factor,sigmaV,sigmaW,max_iter_steps,method = read_parameters(parameters)
    
    print("PARAMETERS : ")
    print("Gamma : {0}, Factor : {1}, SigmaV : {2}, SigmaW : {3}, max_iter_steps : {4}, method : {5}".format(gamma,factor,sigmaV,sigmaW,max_iter_steps,method))
    print()
    resum_optim = {}

    #To save the results
    folder2save+='/'+method+'/'
    try_mkdir(folder2save)
    
    folder_resume_results = folder2save+'/dict_resume_opt/'
    try_mkdir(folder_resume_results)

    decalage =  RawRegistration(template,target, use_torch=False)  

    #vertices
    template = torch.from_numpy(template).clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    target = torch.from_numpy(target).detach().to(dtype=torchdtype, device=torchdeviceId)
    #faces
    template_connections = torch.from_numpy(template_connections).detach().to(dtype=torch.long, device=torchdeviceId)
    target_connections = torch.from_numpy(target_connections).detach().to(dtype=torch.long, device=torchdeviceId)
    
    #The kernel deformation definition
    Kv = Sum4GaussKernel(sigma=sigmaV)

    #Initial Momenta : the optimization variables
    p0 = torch.zeros(template.shape, dtype=torchdtype, device=torchdeviceId, requires_grad=True)

    #Save the rigid deformation and deformed template
    np.save(folder2save +'/rigid_def.npy', template.detach().cpu().numpy())

    for j,sigW in enumerate(sigmaW):
        print("The diffeomorphic part")
        print("Scale : ", sigW)
        tensor_scale = sigW

        ######### Loss to update in both case
        if structure == "Surfaces":
            dataloss_att = SurfacesDataloss(method, template_connections, target, target_connections, tensor_scale).data_attachment()   
        else:
            dataloss_att = CurvesDataloss(method, template_connections, target, target_connections, tensor_scale).data_attachment()   

        ######### The diffeo part
        loss = LDDMMloss(Kv,dataloss_att,sigmaV, gamma=gamma)

        dict_opt_name = 'scale_'+str(int(tensor_scale))+'_sum4kernels_'+str(int(sigmaV.detach().cpu().numpy()))
        p0,nit,total_time = optimization.opt(loss, p0, template, max_iter_steps[j], folder_resume_results, dict_opt_name)
        p_i,deformed_i = Shooting(p0, template, Kv)

        resum_optim[str(sigW)]="Diffeo Nb Iterations : "+str(nit)+", total time : "+str(total_time)

        filesavename = 'Scale_'+str(int(tensor_scale))

        qnp_i = deformed_i.detach().cpu().numpy()

        print("Saving in : ", folder2save)
        template_labels = np.ones(qnp_i.shape[0])
        
        f = folder2save+'/'+filesavename+'.npz'
        np.savez(f, vertices = qnp_i, connections = template_connections.detach().cpu().numpy(), labels = template_labels)
        #export_labeled_vtk(qnp_i,template_connections,template_labels,filesavename,folder2save)

        p0_np = p0.detach().cpu().numpy()

        np.save(folder2save +'/template_' + filesavename +'.npy', template.detach().cpu().numpy())
        np.save(folder2save +'/Momenta_'+filesavename+'.npy',p0_np)
        np.save(folder2save +'/'+filesavename+'.npy',qnp_i)

    #Save the optimization informations
    with open(folder2save+'/resume_optimization.txt',"w") as f:
        f.write(json.dumps(resum_optim))
    f.close()

    np.save(folder2save+'/target.npy',target.detach().cpu().numpy())
    np.save(folder2save+'/momenta2apply.npy',p0_np) 
    np.save(folder2save+'/template2apply.npy',template.detach().cpu().numpy()) 

    return 0


