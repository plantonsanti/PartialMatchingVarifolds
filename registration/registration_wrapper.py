#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys 


current_path = os.path.dirname(__file__)

sys.path.append(os.path.abspath(current_path + "/../utils"))
sys.path.append(os.path.abspath(current_path + "/../registration"))

import optimization

from IO               import *
from registration     import *
from keops_utils      import *

from data_attachment  import DatalossClass

import torch
import numpy as np
import json


def register_structure(template, template_connections,
                        target, target_connections,
                        folder2save,
                        parameters={"default" : True},
                        structure = None, initial_registration = 0, oriented = False):
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

    # Cuda management
    use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod = TestCuda()

    gamma,sigmaV,sigmaW,max_iter_steps,method = read_parameters(parameters)
    
    print("PARAMETERS : ")
    print("Gamma : {0}, SigmaV : {1}, SigmaW : {2}, max_iter_steps : {3}, method : {4}".format(gamma,sigmaV,sigmaW,max_iter_steps,method))
    print()
    resum_optim = {}

    #To save the results
    folder2save+='/'+method+'/'
    try_mkdir(folder2save)
    
    folder_resume_results = folder2save+'/dict_resume_opt/'
    try_mkdir(folder_resume_results)


    if initial_registration == 2:
        decalage =  RegisterRoot(template,target) 
    elif initial_registration == 1:
        decalage =  RawRegistration(template,target, use_torch=False)  

    np.savez(folder2save+'initial_template.npz', vertices = template, 
             connections = template_connections)


    #vertices
    template = torch.from_numpy(template).clone().detach().to(dtype=torchdtype, device=torchdeviceId).requires_grad_(True)
    target = torch.from_numpy(target).detach().to(dtype=torchdtype, device=torchdeviceId)
    #faces
    template_connections = torch.from_numpy(template_connections).detach().to(dtype=torch.long, device=torchdeviceId)
    target_connections = torch.from_numpy(target_connections).detach().to(dtype=torch.long, device=torchdeviceId)
        
    #The kernel deformation definition (Sum of gaussian kernels at 4 different scales)
    Kv = Sum4GaussKernel(sigma=sigmaV)

    #Initial Momenta : the optimization variables
    p0 = torch.zeros(template.shape, dtype=torchdtype, device=torchdeviceId, requires_grad=True)

    #Save the rigid deformation and deformed template
    np.save(folder2save +'/rigid_def.npy', template.detach().cpu().numpy())

    for j,sigW in enumerate(sigmaW):
        print("The diffeomorphic part")
        print("Scale : ", sigW)
        tensor_scale = sigW

        ######### Data loss to compute at each diffeomorphic shooting
        Instance = DatalossClass(method, template_connections, 
                                    target, target_connections, 
                                    tensor_scale, 
                                    source_vertices = template, structure = structure, oriented=oriented)  
        dataloss_att = Instance.data_attachment()   

        ######### The diffeo part
        loss = LDDMMloss(Kv,dataloss_att,sigmaV, gamma=gamma)

        dict_opt_name = 'scale_'+str(int(tensor_scale))+'_sum4kernels_'+str(int(sigmaV.detach().cpu().numpy()))
        p0,nit,total_time = optimization.opt(loss, p0, template, max_iter_steps[j], folder_resume_results, dict_opt_name)
        p_i,deformed_i = Shooting(p0, template, Kv)

        ######### Saving the results 
        resum_optim[str(sigW)]="Diffeo Nb Iterations : "+str(nit)+", total time : "+str(total_time)

        filesavename = 'Scale_'+str(float(tensor_scale.detach().cpu().numpy()))

        qnp_i = deformed_i.detach().cpu().numpy()

        print("Saving in : ", folder2save)
        template_labels = np.ones(qnp_i.shape[0])
        
        f = folder2save+'/'+filesavename+'.npz'
        np.savez(f, points = qnp_i, connections = template_connections.detach().cpu().numpy(), labels = template_labels)

        p0_np = p0.detach().cpu().numpy()
        np.save(folder2save +'/Momenta_'+filesavename+'.npy',p0_np)

    #Save the optimization informations
    with open(folder2save+'/resume_optimization.txt',"w") as f:
        f.write(json.dumps(resum_optim))
    f.close()

    np.savez(folder2save+'target.npz', points = target.detach().cpu().numpy(), 
             connections = target_connections.detach().cpu().numpy())
    
    np.save(folder2save+'/momenta2apply.npy',p0_np) 
    np.save(folder2save+'/control_points.npy',template.detach().cpu().numpy()) 

    return 0




import argparse

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("path_to_source", help="path to the source file, that should be a .npz file containing the points (key 'points'), and their connections (key 'connections').",
                        type=str)
    parser.add_argument("path_to_target", help="path to the target file, that should be a .npz file containing the points (key 'points'), and their connections (key 'connections').",
                        type=str)
    parser.add_argument("path_to_save", help="path to folder in which to save the results.",
                        type=str)
    parser.add_argument("-o", "--oriented", action="store_true", help="use oriented shapes", default=0)
    parser.add_argument("-p", "--parameters", type=str, help="path to config file, a .json containing the LDDMM parameters", default="empty")
    parser.add_argument("-i", "--initial_registration", type=int, choices=[0, 1, 2], 
                        help="(Default) 0: no initial registration \n 1: register the centers of mass \n 2: register the first points", 
                        default=0)
    parser.add_argument("-s", "--structure", type=str, choices=["curves", "surfaces"], 
                        help="The structure of the shapes to align. If None, the structure will be deduced from the connections of the vertices.", 
                        default=None)
    
    args = parser.parse_args()
    
    V_template_np = np.load(args.path_to_source, allow_pickle=True)['points']
    F_template_np = np.load(args.path_to_source, allow_pickle=True)['connections'].astype(dtype = 'int32')             

    V_target_np = np.load(args.path_to_target, allow_pickle=True)['points']
    F_target_np = np.load(args.path_to_target, allow_pickle=True)['connections'].astype(dtype = 'int32')
    
    if args.oriented:
        print("Using oriented shapes")
        
    if args.parameters != "empty" :
        with open(args.parameters,'r') as json_file:
            parameters = json.load(json_file) 
    else:
        parameters={"default" : True}

    args.initial_registration
    if args.initial_registration == 1:
        print("Initial registration of centers of masses")
    elif args.initial_registration == 2:
        print("Initial registration of first points")
        
    out = register_structure(V_template_np, F_template_np,
                             V_target_np, F_target_np,
                             args.path_to_save,
                             parameters=parameters,
                             structure = args.structure, 
                             initial_registration = args.initial_registration, 
                             oriented = args.oriented)

    return out

if __name__ == "__main__":
    sys.exit(main())

