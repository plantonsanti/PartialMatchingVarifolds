#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:30:25 2019

@author: pantonsanti
"""

import torch
import numpy as np

from keops_utils import GaussLinKernel, TestCuda, PartialWeightedGaussLinKernel, OrientedGaussLinKernel, PartialWeightedGaussLinKernelOriented

# Cuda management
use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod = TestCuda(verbose=False)

####################################################################
# Computing the input structures
# ^^^^^^^^^^^^^^^^^^^^

def Compute_structures_surface(V,F):
    """
    For the linear normal kernel
    
    Parameters
    ----------
    @param : V : torch tensor
                 n-points x d-dimension points.
    @param : F : torch Long tensor
                 m-connections x 2-dim tensor containing pair of connected points' indices.

    Returns
    -------
    @output : centers        : torch tensor
                               npoints-1 x d-dimension points, the centers of each face.
    @output : length         : float
                               npoints-1 x 1-dimension tensor, the areas of each face.
    @output : normals        : torch tensor
                               npoints-1 x d-dimension normalized vectors of normals of each face. 
    """
    
    V0, V1, V2 = V.index_select(0, F[:, 0]), V.index_select(0, F[:, 1]), V.index_select(0, F[:, 2])   
    centers, normals =  (V0 + V1 + V2) / 3, .5 * torch.cross(V1 - V0, V2 - V0)    
    length = (normals ** 2).sum(dim=1)[:, None].sqrt()
    
    return centers, length, normals/length

 
#####################################################################
########################### DATALOSSES ##############################
#####################################################################
    
#%% Dataloss for surfaces
accepted_methods = ['Varifold', 'PartialVarifold', 'PartialVarifoldLocal', 'PartialVarifoldLocal2', 'PartialVarifoldNormalizedOriented']

class SurfacesDataloss():
    
    def __init__(self, method, source_faces, target_vertices, target_faces,
                 sigmaW, s_faces_selected = None, t_faces_selected = None):
        """
        Initialize the dataloss function to compare curves or graph of curves. 
        These parameters are shared by all datalosses.
        
        Parameters
        ----------
        @param : method             : string, the name of the attachment called. 
                                      Default : Varifold.
                                      Currently accepted methods : 
                                          Varifold, 
                                          PartialVarifold, 
                                          PartialVarifoldLocal
                                          
        @param : source_faces       : torch Long tensor 
                                      Pairs of integers : face connectivity of source surface.
        @param : target_vertices    : torch tensor
                                      n-points x d-dimension vertices of the target.
        @param : target_faces       : torch Long tensor 
                                      Pairs of integers : face connectivity of target surface.
        @param : sigmaW             : torch float tensor, 
                                      positive scale of the data attachment term.

        @param  : s_faces_selected    : torch Long tensor, 
                                      The selected face connectivities of source surface. Default : None.
                                      If s_faces_selected is not None, the data attachment will be computed on a subset of the source face connectivities.
        @param  : t_faces_selected    : torch Long tensor, 
                                      The selected face connectivities of target surface. Default : None.
                                      If t_faces_selected is not None, the data attachment will be computed on a subset of the target face connectivities.    
        """

        self.selected_source = True
        #If no selected connections provided, set selected to whole connections:
        if s_faces_selected is None:  
            s_faces_selected = source_faces
            self.selected_source = False
        if t_faces_selected is None:  
            t_faces_selected = target_faces   
    
        self.FS     = source_faces
        self.FS_sel = s_faces_selected
        
        self.FT     = target_faces
        self.FT_sel = t_faces_selected
        self.VT     = target_vertices
        
        print(self.FS)

        self.sigmaW = sigmaW
        self.method = method
    
    
    def set_method(self,method):
        """
        Parameters
        ----------
        @param : method : string, the name of the attachment called. 
                            Default : 'Varifold'.
                            Currently accepted methods : 
                                Varifold, 
                                PartialVarifold, 
                                PartialVarifoldLocal
        """

        if method in accepted_methods:
            self.method = method
        else:
            print('The required method {0} is not accepted, please use one of : {1}'.format(method,accepted_methods))
            print('Using default : Varifold')
            self.method = 'Varifold'

        return


    #Function to regroup the data attachment calling.
    def data_attachment(self):
        """
        Return the called dataloss function depending on the initialized method.                                   
                                              
        Returns
        -------                                      
        @output : dataloss  : function of the source tree points position. 
        """
        
        if self.method == "PartialVarifold": 
            print("Using Partial Varifold")
            dataloss = self.PartialVarifoldSurface()
    
        elif self.method == "PartialVarifoldLocal": 
            print("Using Partial Varifold Local version")
            dataloss = self.PartialVarifoldSurfaceLocal()
    
        elif self.method == "PartialVarifoldLocal2": 
            print("Using Partial Varifold Local version 2")
            dataloss = self.PartialVarifoldLocal2()
        
        elif self.method == "PartialVarifoldNormalizedOriented": 
            print("Using Partial Varifold Local version 2")
            dataloss = self.PartialVarifoldLocalNormalizedOriented()

        else:
            if(self.method!="Varifold"):
                print("Specified method not accepted, using default data attachment : Varifold")
            print("Using Varifold")
            dataloss = self.VarifoldSurface()
    
        return dataloss
    
    
    #%% Varifold datalosses
    def VarifoldSurface(self):
        """
        The default dataloss : Varifold. 
        
        Returns
        -------
        @output : loss : the data attachment function.
        """
        K = GaussLinKernel(sigma=self.sigmaW)
    
        CT, LT, NTn = Compute_structures_surface(self.VT, self.FT_sel)
    
        cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()
        
        def loss(VS):
            CS, LS, NSn = Compute_structures_surface(VS, self.FS_sel)
            cost = 10*np.pi**2/4*(cst + (LS * K(CS, CS, NSn, NSn, LS)).sum() - 2 * (LS * K(CS, CT, NSn, NTn, LT)).sum())
            return cost/(self.sigmaW**2) 
        return loss


    def PartialVarifoldSurface(self):
        """
        The Partial Varifold, in its global version. Designed to include the 
        deformed source into the target.  
        
        Returns
        -------
        @output : loss : the data attachment function.
        """
        K = GaussLinKernel(sigma=self.sigmaW)
    
        CT, LT, NTn = Compute_structures_surface(self.VT, self.FT_sel)

        def loss(VS):
            CS, LS, NSn = Compute_structures_surface(VS, self.FS_sel)
            cross = (LS * K(CS, CT, NSn, NTn, LT)).sum()
            cost_S = (LS * K(CS, CS, NSn, NSn, LS)).sum() - cross
            return cost_S*cost_S/(self.sigmaW**2) 
            
        return loss
    
    
    def PartialVarifoldSurfaceLocal(self):
        """
        The Partial Varifold, in its local version. Designed to include the 
        deformed source into the target. 
        
        Returns
        -------
        @output : loss : the data attachment function.
        """
        def g2(x):
            return (x**2)*(x>0).float()
    
        K = GaussLinKernel(sigma=self.sigmaW)
    
        CT, LT, NTn = Compute_structures_surface(self.VT, self.FT_sel)
    
        def far_pen(xs):
            return 1./(xs+1)
        
        def loss(VS):
            CS, LS, NSn = Compute_structures_surface(VS, self.FS_sel)            
            xs = K(CS, CT, NSn, NTn, LT)

            cost = (LS * g2( K(CS, CS, NSn, NSn, LS) - xs)).sum() 

            return cost/(self.sigmaW**2) 
        return loss
    

    def PartialVarifoldLocal2(self):
        """
        The Partial Weighted Varifold, in its local version. Designed to include
        the deformed source into the target. 
        
        Returns
        -------
        @output : loss : the data attachment function.
        """

        def g2(x):
            return (x**2)*(x>0).float()

        K = GaussLinKernel(sigma=self.sigmaW)
    
        CT, LT, NTn = Compute_structures_surface(self.VT, self.FT_sel)
        omega_T = K(CT, CT, NTn, NTn, LT)
    
        def far_pen(xs):
            return 1./(xs+1)
        
        WeightedKernel = PartialWeightedGaussLinKernel(sigma=self.sigmaW)

        def loss(VS):
            CS, LS, NSn = Compute_structures_surface(VS, self.FS_sel)

            omega_S    = K(CS, CS, NSn, NSn, LS)
            omega_tild = WeightedKernel(CS, CT, NSn, NTn, omega_S, omega_T, LT)
            
            cost = ( LS * g2( omega_S - omega_tild )).sum()
            return cost/(self.sigmaW**2) 
        return loss
   

    def PartialVarifoldLocalNormalizedOriented(self):
        """
        The Partial Weighted Varifold, in its local version. Designed to include
        the deformed source into the target. 
        
        Returns
        -------
        @output : loss : the data attachment function.
        """

        def g2(x):
            return (x**2)*(x>0).float()
    
        K = OrientedGaussLinKernel(sigma=self.sigmaW)
        WeightedKernel = PartialWeightedGaussLinKernelOriented(sigma=self.sigmaW)

        CT, LT, NTn = Compute_structures_surface(self.VT, self.FT_sel)
        omega_T = K(CT, CT, NTn, NTn, LT)

        def loss(VS):
            CS, LS, NSn = Compute_structures_surface(VS, self.FS_sel)

            omega_S    = K(CS, CS, NSn, NSn, LS)
            omega_tild = WeightedKernel(CS, CT, NSn, NTn, omega_S, omega_T, LT)
            
            cost = ( LS * g2( omega_S - omega_tild  )).sum()

            return cost/(self.sigmaW**2)
        return loss 
    
    
   
