#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:30:25 2019

@author: pantonsanti
"""

# Standard imports
from keops_utils import TestCuda
import torch
from torch.autograd import grad
import numpy as np

use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod = TestCuda(verbose=False)


# Barycenter registration
def RawRegistration(VS,VT, use_torch = True):
    """
    A simple registration of the barycenters.
    Translate the source. 

    @param : VS : (n_source x dim) torch tensor, the source vertices (that are translated in the procedure)
    @param : VT : (n_target x dim) torch tensor, the target vertices

    @output : decalage : (1 x dim) torch tensor, the translation applied along each axis.  
    """

    if not use_torch :
        bary_S = np.mean(VS,0)
        bary_T = np.mean(VT,0)
    else:
        bary_S = torch.mean(VS,0)
        bary_T = torch.mean(VT,0)
    decalage = bary_T - bary_S

    for k in range(VS.shape[0]):
        VS[k,:]+=decalage

    return decalage



####################################################################
#%% Custom ODE solver, for ODE systems which are defined on tuples
def RalstonIntegrator(nt=10):
    def f(ODESystem, x0, deltat=1.0):
        x = tuple(map(lambda x: x.clone(), x0))
        dt = deltat / nt
        for i in range(nt):
            xdot = ODESystem(*x)
            xi = tuple(map(lambda x, xdot: x + (2 * dt / 3) * xdot, x, xdot))
            xdoti = ODESystem(*xi)
            x = tuple(map(lambda x, xdot, xdoti: x + (.25 * dt) * (xdot + 3 * xdoti), x, xdot, xdoti))
        return x
    
    return f


#####################################################################
#%% Hamiltonian system

def Hamiltonian(K):
    def H(p, q):
        return .5 * (p * K(q, q, p)).sum()
    return H


def HamiltonianSystem(K):
    H = Hamiltonian(K)
    def HS(p, q):
        Gp, Gq = grad(H(p, q), (p, q), create_graph=True)
        return -Gq, Gp
    return HS


#####################################################################
#%% Shooting approach
def Shooting(p0, q0, K, deltat=1.0, Integrator=RalstonIntegrator()):
    return Integrator(HamiltonianSystem(K), (p0, q0), deltat)

def Flow(x0, p0, q0, K, deltat=1.0, Integrator=RalstonIntegrator()):
    HS = HamiltonianSystem(K)
    def FlowEq(x, p, q):
        return (K(x, q, p),) + HS(p, q) #concatenation des fonctions pour le système hamiltonnien
    return Integrator(FlowEq, (x0, p0, q0), deltat)


def ShootingTraj(p0, q0, K, deltat=1.0,nt=30):
    """
    Compute the intermediate positions of the source, given the initial momentum p0 
    """
    HS = HamiltonianSystem(K)
    x_traj,x = [],q0
    p = p0
    dt = deltat/nt
    for i in range(nt):
        pdot,xdot = HS(p,x)
        x,p = x+dt*xdot,p+dt*pdot
        x_traj.append(x)
    return x_traj


#####################################################################
#%% The functions to optimize in a shooting algorithm
def LDDMMloss(K, dataloss,sigmaV, gamma=0.1):
    def loss(p0,q0):
        p,q = Shooting(p0, q0, K)
        Ev = gamma*Hamiltonian(K)(p0, q0)/(sigmaV**2)
        A = dataloss(q)
        print('Energy : ', Ev, ' Data attachment : ', A)
        return gamma, Ev, A
        
    return loss


def LDDMMloss_ctrlpts(K, dataloss,sigmaV, gamma=0.1):
    def loss(x0,p0,q0):
        x,p,q = Flow(x0, p0, q0, K)
        return gamma*Hamiltonian(K)(p0, q0)/(sigmaV**2) + dataloss(x)
        
    return loss
