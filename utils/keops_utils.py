#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:30:25 2019

@author: pantonsanti
"""

import torch
from pykeops.torch import LazyTensor, Genred, Vi, Vj

################### GPU management #########################"
# torch type and device
def TestCuda(verbose = True):
    use_cuda = torch.cuda.is_available()
    if(verbose):    
        print("Use cuda : ",use_cuda)
    torchdeviceId = torch.device('cuda:0') if use_cuda else 'cpu'
    torchdtype = torch.float32
    KernelMethod = 'CPU'
    
    if(use_cuda):
        torch.cuda.set_device(torchdeviceId)
        KernelMethod = 'auto'
    # PyKeOps counterpart
    KeOpsdeviceId = torchdeviceId.index  # id of Gpu device (in case Gpu is  used)
    KeOpsdtype = torchdtype.__str__().split('.')[1]  # 'float32'

    #print(KeOpsdtype)
    return use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod

use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod = TestCuda(verbose=False)


############################################################
# function to transfer data on Gpu only if we use the Gpu
def CpuOrGpu(x):
    if use_cuda:
        if type(x)==tuple:
            x = tuple(map(lambda x:x.cuda(device=torchdeviceId),x))
        else:
            x = x.cuda(device=torchdeviceId)
    return x



###################################################################
# Define "Gaussian-CauchyBinet" kernel :math:`(K(x,y,u,v)b)_i = \sum_j \exp(-\|x_i-y_j\|^2) \langle u_i,v_j\rangle^2 b_j`

def GaussLinKernel(sigma):

    x, y, u, v, b = Vi(0,3), Vj(1,3), Vi(2,3), Vj(3,3), Vj(4,1)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2*gamma).exp() * (u*v).sum()**2

    return (K*b).sum_reduction(axis=1)


def OrientedGaussLinKernel(sigma):

    x, y, u, v, b = Vi(0,3), Vj(1,3), Vi(2,3), Vj(3,3), Vj(4,1)
    gamma = 1 / (sigma * sigma)
    D2 = x.sqdist(y)
    K = (-D2*gamma).exp() * ((u*v).sum()).exp()

    return (K*b).sum_reduction(axis=1)


def SobolevKernel():
    x, y = Vi(0,3), Vj(1,3)
    D2 = x.sqdist(y)
    K = D2.pow(3/2)
    return (K).sum_reduction(axis=1)


def GenericGaussKernel(ref_scale,list_coefs = [1]):
    """

    Given a deformation scale for the kernels, generates the sum of the kernels at the scale divided by the coeff. 
    

    Parameters
    ----------
    ref_scale : torch tensor
        The scale of the deformations kernels (that is to be divided by the coefficients of list_coefs)
    list_coefs : list of floats
        The coefficients that will divide the deformations scale and ponder the kernels. 

    Returns
    -------
    K : kernel -or sum of kernels- funtion.

    """
    
    def K(x,y,b = None):
        """
        The kernel function. 

        Parameters
        ----------
        x,y : torch tensor
            The points x.shape = (n_pts_x, dim) y.shape = (n_pts_y, dim)
        b (Optional) : torch tensor
            Optional momenta or weights. Default: None 

        Returns
        -------
        a_i : LazyTensor

        If b is None, a_i = sum_j exp(-|x_i-y_j|^2)
        else,         a_i = sum_j exp(-|x_i-y_j|^2).b_i
 
        """

        x_i = LazyTensor( x[:,None,:] )  # x_i.shape = (n_pts_x, 1, dim)
        y_j = LazyTensor( y[None,:,:] )  # y_j.shape = ( 1, n_pts_y,dim)

        D_ij = ((x_i - y_j)**2).sum(dim=2)  # Symbolic (n_pts_x,n_pts_y,1) matrix of squared distances
        
        ref_scale2 = 1/(ref_scale**2)

        weighting = 1/len(list_coefs)

        for i,coef in enumerate(list_coefs):

            c2 = coef**2*ref_scale2
            
            if i==0:
                K_ij =  weighting*(- D_ij*c2).exp()  #  # Symbolic (n_pts_x,n_pts_y,1) matrix
            else:
                K_ij += weighting*(- D_ij*c2).exp()  #  # Symbolic (n_pts_x,n_pts_y,1) matrix

        if b is None:
            a_i = K_ij.sum(axis=1)

        else:
            b =   LazyTensor( b[None,:,:] )    
            a_i = (K_ij*b).sum(axis=1)

        return a_i

    return K


def PartialWeightedGaussLinKernel(sigma, epsilon = 0.0001):
    """

    Given a deformation scale for the kernels, generates the sum of the kernels at the scale divided by the coeff. 
    

    Parameters
    ----------
    ref_scale : torch tensor
        The scale of the deformations kernels (that is to be divided by the coefficients of list_coefs)
    list_coefs : list of floats
        The coefficients that will divide the deformations scale and ponder the kernels. 

    Returns
    -------
    K : kernel -or sum of kernels- funtion.

    """
    
    gamma = 1/sigma**2

    def min_x1(x):
        return .5*(x+1-(epsilon+(x-1)**2).sqrt())

    

    def K(x, y, u, v, omega_x, omega_y, by):
        """
        The kernel function. 

        Parameters
        ----------
        x,y : torch tensor
            The points x.shape = (n_pts_x, dim) y.shape = (n_pts_y, dim)
        b (Optional) : list of integers
            Optional momenta. Default: None 

        Returns
        -------
        a_i : LazyTensor

        If b is None, a_i = sum_j exp(-|x_i-y_j|^2)
        else,         a_i = sum_j exp(-|x_i-y_j|^2).b_i
 
        """

        x_i = LazyTensor( x[:,None,:] )  # x_i.shape = (n_pts_x, 1, dim)
        y_j = LazyTensor( y[None,:,:] )  # y_j.shape = ( 1, n_pts_y,dim)

        b_j = LazyTensor( by[None,:] )

        u_i = LazyTensor( u[:,None,:] )
        v_j = LazyTensor( v[None,:,:] )

        D_ij = ((x_i - y_j)**2).sum(dim=2)  # Symbolic (n_pts_x,n_pts_y,1) matrix of squared distances
        omega_ij = (- D_ij*gamma).exp() * (u_i*v_j).sum()**2

        M_compare = LazyTensor(omega_x[:,None,:])*LazyTensor((1/omega_y)[None,:,:])
        matrix_weights = min_x1(M_compare)

        omega_weighted_ij = (matrix_weights*omega_ij*b_j).sum(dim=1) #(n_pts_y,n_pts_y) LAZZY TENSOR

        return omega_weighted_ij

    return K


def PartialWeightedGaussKernel(sigma, epsilon = 0.0001):
    """

    Given a deformation scale for the kernels, generates the sum of the kernels at the scale divided by the coeff. 
    

    Parameters
    ----------
    ref_scale : torch tensor
        The scale of the deformations kernels (that is to be divided by the coefficients of list_coefs)
    list_coefs : list of floats
        The coefficients that will divide the deformations scale and ponder the kernels. 

    Returns
    -------
    K : kernel -or sum of kernels- funtion.

    """
    
    gamma = 1/sigma**2

    def min_x1(x):
        return .5*(x+1-(epsilon+(x-1)**2).sqrt())

    def K(x, y, omega_x, omega_y, by):
        """
        The kernel function. 

        Parameters
        ----------
        x,y : torch tensor
            The points x.shape = (n_pts_x, dim) y.shape = (n_pts_y, dim)
        b (Optional) : list of integers
            Optional momenta. Default: None 

        Returns
        -------
        a_i : LazyTensor

        If b is None, a_i = sum_j exp(-|x_i-y_j|^2)
        else,         a_i = sum_j exp(-|x_i-y_j|^2).b_i
 
        """

        x_i = LazyTensor( x[:,None,:] )  # x_i.shape = (n_pts_x, 1, dim)
        y_j = LazyTensor( y[None,:,:] )  # y_j.shape = ( 1, n_pts_y,dim)

        b_j = LazyTensor( by[None,:] )

        D_ij = ((x_i - y_j)**2).sum(dim=2)  # Symbolic (n_pts_x,n_pts_y,1) matrix of squared distances
        omega_ij = (- D_ij*gamma).exp()

        M_compare = LazyTensor(omega_x[:,None,:])*LazyTensor((1/omega_y)[None,:,:])
        matrix_weights = min_x1(M_compare)

        omega_weighted_ij = (matrix_weights*omega_ij*b_j).sum(dim=1) #(n_pts_y,n_pts_y) LAZZY TENSOR

        return omega_weighted_ij

    return K


def PartialWeightedGaussLinKernelOriented(sigma, epsilon = 0.0001):
    """

    Given a deformation scale for the kernels, generates the sum of the kernels at the scale divided by the coeff. 
    

    Parameters
    ----------
    ref_scale : torch tensor
        The scale of the deformations kernels (that is to be divided by the coefficients of list_coefs)
    list_coefs : list of floats
        The coefficients that will divide the deformations scale and ponder the kernels. 

    Returns
    -------
    K : kernel -or sum of kernels- funtion.

    """
    
    gamma = 1/sigma**2

    def min_x1(x):
        return .5*(x+1-(epsilon+(x-1)**2).sqrt())

    def K(x, y, u, v, omega_x, omega_y, by):
        """
        The kernel function. 

        Parameters
        ----------
        x,y : torch tensor
            The points x.shape = (n_pts_x, dim) y.shape = (n_pts_y, dim)
        b (Optional) : list of integers
            Optional momenta. Default: None 

        Returns
        -------
        a_i : LazyTensor

        If b is None, a_i = sum_j exp(-|x_i-y_j|^2)
        else,         a_i = sum_j exp(-|x_i-y_j|^2).b_i
 
        """

        x_i = LazyTensor( x[:,None,:] )  # x_i.shape = (n_pts_x, 1, dim)
        y_j = LazyTensor( y[None,:,:] )  # y_j.shape = ( 1, n_pts_y,dim)

        b_j = LazyTensor( by[None,:] )

        u_i = LazyTensor( u[:,None,:] )
        v_j = LazyTensor( v[None,:,:] )

        D_ij = ((x_i - y_j)**2).sum(dim=2)  # Symbolic (n_pts_x,n_pts_y,1) matrix of squared distances
        omega_ij = (- D_ij*gamma).exp() * ((u_i*v_j).sum()).exp()

        M_compare = LazyTensor(omega_x[:,None,:])*LazyTensor((1/omega_y)[None,:,:])
        matrix_weights = min_x1(M_compare)

        omega_weighted_ij = (matrix_weights*omega_ij*b_j).sum(dim=1) #(n_pts_y,n_pts_y) LAZZY TENSOR

        return omega_weighted_ij

    return K


###############################################################
################# SOME WRAPPING FUNCTIONS #####################
###############################################################

def Sum3GaussKernel(sigma = 100):

    list_coefs = [1., 2., 4.] 
    K = GenericGaussKernel(sigma,list_coefs)

    return K

def Sum4GaussKernel(sigma = 100):

    list_coefs = [1., 4., 8., 16.] 
    K = GenericGaussKernel(sigma,list_coefs)

    return K


def Sum4GaussKernel_bis(sigma = 100):

    list_coefs = [1., 2., 4., 8.] 
    K = GenericGaussKernel(sigma,list_coefs)

    return K

