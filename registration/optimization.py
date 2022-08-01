#!/usr/bin/env python3
# -*- coding: utf-8 -*-

######################### perform optimization ##############################
import torch
import time
import pickle
import os
import numpy as np
from scipy.optimize import minimize

params_opt=dict({"lr" : 1,"maxcor" : 10, "gtol" : 1e-6, "tol" : 1e-6, "use_scipy" : True, "method" : 'L-BFGS-B'})


def opt(loss,p0,q0, maxiter = 100, folder2save = '',savename = ''):
    """
    Optimization function calling either scipy or torch method. 
    p0 is the variable to optimize, and can either be the initial momenta or a quaternion depending on the deformation one want to implement.
    """
    lr        = params_opt["lr"] 
    maxcor    = params_opt["maxcor"]
    gtol      = params_opt["gtol"]
    tol       = params_opt["tol"]
    use_scipy = params_opt["use_scipy"] #If use_scipy : perform otpimization with LBFGS on scipy. 
    method    = params_opt["method"]
    options   = dict( maxiter = maxiter,
                      ftol    = tol,
                      gtol    = gtol,
                      maxcor  = maxcor        # Number of previous gradients used to approximate the Hessian
                )

    loss_dict = {}

    loss_dict['A'] = [0]
    loss_dict['E'] = [0]

    optimizer = torch.optim.LBFGS([p0])
    start = time.time()
    print('performing optimization...')
    opt.nit = -1
    def closure():
        opt.nit += 1; it = opt.nit

        optimizer.zero_grad()
        gamma,E,A = loss(p0,q0)

        L = gamma*E+A
        
        L.backward(retain_graph=True) #ATTENTION, CHANGE POUR ENCHAINER APRES RIGIDE, SINON ENLEVER RETAIN GRAPH !!! 

        print("Iteration ",it)

        if(folder2save != ''):
            if(opt.nit % 5 == 0):
                loss_dict['A'].append(float(A.detach().cpu().numpy()))
                loss_dict['E'].append(float(E.detach().cpu().numpy()))

        return L

    # Optimisation using scipy : we need to transfer the data from variable to float64
    def numpy_closure(vec):
        vec = lr*vec.astype('float64')
        numpy_to_model(p0,vec)
        c =  closure().data.view(-1).cpu().numpy()[0]
        dvec = model_to_numpy(p0,grad = True)
        return (c,dvec)
    
    def model_to_numpy(p, grad=False) :
        if grad :
            tensors = p.grad.data.view(-1).cpu().numpy()
        else :
            tensors = p.data.view(-1).cpu().numpy()     
        return np.ascontiguousarray( np.hstack(tensors) , dtype='float64' )
    
    def numpy_to_model(p, vec) :
        p.data = torch.from_numpy(vec).view(p.data.size()).type(p.data.type())

    if use_scipy :
        res = minimize( numpy_closure,      # function to minimize
                model_to_numpy(p0), # starting estimate
                method  = method,
                jac     = True,             # matching_problems also returns the gradient
                options = options)
        print(res.message)
        
    else :
        for i in range(int(maxiter/20)+1):            # Fixed number of iterations
            optimizer.step(closure)         # "Gradient descent" step.
    
    total_time = round(time.time()-start,2)
    print('Optimization time : ',total_time,' seconds')

    if(folder2save != ''):
        try:
            os.mkdir(folder2save)
        except OSError:
            pass

        loss_dict['Time'] = total_time
        loss_dict['it'] = opt.nit
        
        with open(folder2save+'/dict_'+savename+'.pkl','wb') as f:
            pickle.dump(loss_dict,f)
    return (p0,opt.nit,total_time)
