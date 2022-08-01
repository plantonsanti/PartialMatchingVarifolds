#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 11:30:25 2019

@author: pantonsanti
"""


import sys 
import os
sys.path.append(os.path.abspath("../utils"))

import torch
from torch.autograd import Variable
import numpy as np

from keops_utils import GaussLinKernel, TestCuda, PartialWeightedGaussLinKernel, OrientedGaussLinKernel, PartialWeightedGaussLinKernelOriented, GenericGaussKernel

# Cuda management
use_cuda,torchdeviceId,torchdtype,KeOpsdeviceId,KeOpsdtype,KernelMethod = TestCuda(verbose=False)


####################################################################
# Computing the input structures
# ^^^^^^^^^^^^^^^^^^^^
def Compute_lengths(V,F):
    """
    Parameters
    ----------
    @param : V : torch tensor
                 n-points x d-dimension points.
    @param : F : torch Long tensor
                 m-connections x 2-dim tensor containing pair of connected points' indices.

    Returns
    -------
    @output : average : float
                       Average euclidean distance between connected points in the tree.
    @output : std     : float
                       Standard deviation of the euclidean distance between connected points in the tree.
    """
    V0, V1 = V.index_select(0,F[:,0]), V.index_select(0,F[:,1])
    u                                    =    (V1-V0)
    lengths                              =    (u**2).sum(1)[:, None].sqrt()


    LS = torch.zeros(V.shape[0], dtype=torchdtype, device=torchdeviceId)

    LS[F[:,0]]+= lengths[:,0]
    LS[F[:,1]]+=lengths[:,0]

    return (.5*LS).view(-1,1)


def Compute_structures_curve(V,F):
    """
    For the linear normal kernel
    
    Parameters
    ----------
    @param : V : torch tensor
                 (n,d) points.
    @param : F : torch Long tensor
                 (m,2) tensor containing pair of connected points' indices.

    Returns
    -------
    @output : centers        : torch tensor
                               npoints-1 x d-dimension points, the centers of each discretization segment in the tree.
    @output : lengths        : float
                               npoints-1 x 1-dimension tensor, the length of each discretization segment in the tree.
    @output : normalized_seg : torch tensor
                               npoints-1 x d-dimension normalized vectors of the discretization segments in tne tree. 
    """
    
    V0, V1 = V.index_select(0,F[:,0]), V.index_select(0,F[:,1])
    u                                    =    (V1-V0)
    lengths                              =    (u**2).sum(1)[:, None].sqrt()

    normalized_seg                       =     u / (lengths.view(-1,1)+1e-5) #to avoid dividing by 0
    centers                              =     0.5*(V1+V0)  
    
    return centers, lengths, normalized_seg


def Compute_structures_surface(V,F):
    """
    For the linear normal kernel
    
    Parameters
    ----------
    @param : V : torch tensor
                 (n,d) points.
    @param : F : torch Long tensor
                 (m,3) tensor containing pair of connected points' indices.

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


def Compute_structures(V,F,structure):
    
    if structure == "curves": 
        return Compute_structures_curve(V,F)
    elif structure == "surfaces":
        return Compute_structures_surface(V,F)
    else:
        print("Unrecognized structure, should either be curves or surfaces. Returning 0")
        return 0 


def Compute_sum_normalized_edges_at_vertex(V,F):
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
    @output : centers : torch tensor
                        npoints-1 x d-dimension points, the sum of the oriented discretization segments at each vertex.
    """     
    V0, V1 = V.index_select(0,F[:,0]), V.index_select(0,F[:,1])
    u                     =    (V1-V0)
    lengths               =    (u**2).sum(1).sqrt()
    Normalized_segments   =      u / (lengths.view(-1,1)+1e-5)
    if use_cuda:
        sum_normalizedsegments_atvertex = Variable(torch.cuda.FloatTensor(np.zeros(V.shape)))
    else:
        sum_normalizedsegments_atvertex = Variable(torch.FloatTensor(np.zeros(V.shape)))
    
    sum_normalizedsegments_atvertex = sum_normalizedsegments_atvertex.index_add(0,F[:,0],Normalized_segments)
    sum_normalizedsegments_atvertex = sum_normalizedsegments_atvertex.index_add(0,F[:,1],-Normalized_segments)
    
    return sum_normalizedsegments_atvertex


def Count_Vertex_Neighb(F):
    """
    Number of neighbors for each vertex in the selected subtree.
    Default 
    WARNING : it assumes that the trees are binary trees !
    
    Parameters
    ----------
    @param : F : torch Long tensor
                 m-connections x 2-dim tensor containing pair of connected points' indices.

    Returns
    -------
    @output : neighbors : torch float tensor
                          Number of connected points per vertice in the tree.
    """

    if use_cuda:
        neighbors      =   Variable(torch.cuda.FloatTensor(np.zeros(F.shape[0]+1))) #!! here is the assumption, otherwise it could not be the right size.
    else:
        neighbors      =   Variable(torch.FloatTensor(np.zeros(F.shape[0]+1)))
    
    for ind,val in enumerate(neighbors):
        if(ind in F[:,:]):
            val += (F[:,0]==ind).sum()
            val += (F[:,1]==ind).sum() 

    return neighbors[:, None]


def Select_extrema_connections(F):
    """
    Function to find the end points and bifurcations in a tree.
    
    Parameters
    ----------
    @param : F     : torch Long tensor
                     m-connections x 2-dim tensor containing pair of connected points' indices.
                     
    @param : F_sel : torch Long tensor
                     m-connections x 2-dim tensor containing pair of connected points' indices in a selected subtree.         
                     
    Returns
    -------
    @output : F_selected : torch Long tensor
                           m-connections x 2-dim tensor containing pair of connected points' indices in a selected subtree.  
    """

    nb_neighb = Count_Vertex_Neighb(F)
    
    if use_cuda:
        mask_select      =  torch.cuda.ByteTensor(np.zeros(F.shape))
    else:
        mask_select      =   torch.ByteTensor(np.zeros(F.shape))
    
    for ind,n_neighb in enumerate(nb_neighb): #find the singular points
        if(n_neighb!=2 and n_neighb != 0):
            mask_select[F[:,:]==ind]=1
            
    for ind in range(mask_select.shape[0]): 
        if(mask_select[ind,0]==1):
            mask_select[ind,1] = 1
        if(mask_select[ind,1]==1):
            mask_select[ind,0] = 1
    
    F_select = F[mask_select[:,0],:].unique(dim=0) #select the connections
        
    return F_select


#####################################################################
# Some metric over the curves
def points_spread(V,F):
    """
    Parameters
    ----------
    @param : V : torch tensor
                 n-points x d-dimension points.
    @param : F : torch Long tensor
                 m-connections x 2-dim tensor containing pair of connected points' indices.

    Returns
    -------
    @output : average : float
                       Average euclidean distance between connected points in the tree.
    @output : std     : float
                       Standard deviation of the euclidean distance between connected points in the tree.
    """
    centers, lengths, normalized_seg = Compute_structures_curve(V,F)
    
    average = torch.mean(lengths)
    std = torch.std(lengths)
    
    return average,std

#####################################################################
########################### DATALOSSES ##############################
#####################################################################
#%% Dataloss for curves
accepted_methods = ['Varifold', 
                    'Varifold_Regulocal', 
                    'PartialVarifold', 
                    'PartialVarifoldLocal', 
                    'PartialVarifoldNormalized', 
                    'PartialVarifoldNormalized_Regulocal',
                    'ConstantNormalCycle',
                    'LinearNormalCycle',
                    'CombinedNormalCycle']

dict_accepted_methods = {}
for i, m in enumerate(accepted_methods):
    dict_accepted_methods[i] = m 

class DatalossClass():
    
    def __init__(self, method, source_connections, target, target_connections,
                 sigmaW, s_con_selected = None, t_con_selected = None, source_vertices = None, structure = None, oriented = True):
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
                                          PartialVarifoldLocal, 
                                          PartialVarifoldLocal2, 

                                          
        @param : source_connections : torch Long tensor 
                                      Pairs of integers : the source tree connected points' indices.
        @param : target             : torch tensor
                                      n-points x d-dimension vertices of the target.
        @param : target_connections : torch Long tensor 
                                      Pairs of integers : the target tree connected points' indices.
        @param : sigmaW             : torch float tensor, 
                                      positive scale of the data attachment term.

        @param  : s_con_selected    : torch Long tensor, 
                                      The selected connections in the source tree. Default : None.
                                      If s_con_selected is not None, the data attachment will be computed on a subtree of the source tree.
        @param  : t_con_selected    : torch Long tensor, 
                                      The selected connections in the target tree. Default : None.
                                      If FS_sel is not None, the data attachment will be computed on a subtree of the target tree.    
        """


        if structure is None:
            if target_connections.shape[1] == 2:
                print("Target connections is (n,2), using curves structure")
                self.structure = "curves"
            elif target_connections.shape[1] == 3:
                print("Target connections is (n,2), using curves structure")
                self.structure = "surfaces"
            else:
                print("structure parameter was None, trying to read data but unrecognized target connections shape (should either be (n,2) or (n,3)).")
                print("Unrecognized structure, returning 0")
                return 0
        elif structure in ["curves", "surfaces"]:
            self.structure = structure
        else:
            print("Unrecognized structure (should either be curves or surfaces), returning 0")
            return 0
        self.selected_source = True
        #If no selected connections provided, set selected to whole connections:
        if s_con_selected is None:  
            s_con_selected = source_connections
            self.selected_source = False
        if t_con_selected is None:  
            t_con_selected = target_connections   
    
        self.FS     = source_connections
        self.FS_sel = s_con_selected
        self.VS     = source_vertices
        
        self.FT     = target_connections
        self.FT_sel = t_con_selected
        self.VT     = target
        
        self.oriented = oriented
        
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
                                PartialVarifoldLocal, 
                                PartialVarifoldLocal2, 
        """

        if method in accepted_methods:
            self.method = method
        else:
            print('The required method {0} is not accepted, please use one of : {1}'.format(method,accepted_methods))
            print('Using default : Varifold')
            self.method = 'Varifold'

        return


    #Function to regroup the data attachment call.
    def data_attachment(self):
        """
        Return the called dataloss function depending on the initialized method.                                   
                                              
        Returns
        -------                                      
        @output : dataloss  : function of the source tree points position. 
        """
        
        if self.method == "ConstantNormalCycle":
            print("Constant Normal Cycles")
            dataloss = self.loss_NC_constant() 
                
        elif self.method == "LinearNormalCycle":   
            print("Using Linear Normal Cycles")     
            dataloss = self.loss_NC_linear()
    
        elif self.method == "CombinedNormalCycle": 
            dataloss_cst = self.loss_NC_constant() 
            dataloss_lin = self.loss_NC_linear()
            print("Using Linear Normal Cycles + Constant Normal Cycles")
            def dataloss(source):
                return 1/2*(dataloss_cst(source)+dataloss_lin(source))
    
        elif self.method == "PartialVarifold": 
            print("Using Partial Varifold")
            dataloss = self.PartialVarifold()
    
        elif self.method == "PartialVarifoldLocal": 
            print("Using Partial Varifold Local version")
            dataloss = self.PartialVarifoldLocal()
    
        elif self.method == "PartialVarifoldNormalized": 
            print("Using Partial Oriented Varifold, Normalized Local version")
            dataloss = self.PartialVarifoldLocalNormalized()

        elif self.method == "PartialVarifoldNormalized_Regulocal":
            dataloss = self.PartialVarifoldNormalized_Regulocal()
            
        elif self.method == "Varifold_Regulocal":
            dataloss = self.Varifold_Regulocal()

        else:
            if(self.method!="Varifold"):
                print("Specified method not accepted, using default data attachment : Varifold")
            print("Using Varifold")
            dataloss = self.Varifold()
    
        return dataloss


    #%% Varifold datalosses
    def Varifold(self):
        """
        The default dataloss : Varifold. 
        
        Returns
        -------
        @output : loss : the data attachment function.
        """
        if self.oriented:
            K = OrientedGaussLinKernel(sigma=self.sigmaW)
        else:
            K = GaussLinKernel(sigma=self.sigmaW)
    
        CT, LT, NTn = Compute_structures(self.VT, self.FT_sel, self.structure)
    
        cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()
        
        def loss(VS):
            CS, LS, NSn = Compute_structures(VS, self.FS_sel, self.structure)
            cost = 10*np.pi**2/4*(cst + (LS * K(CS, CS, NSn, NSn, LS)).sum() - 2 * (LS * K(CS, CT, NSn, NTn, LT)).sum())
            return cost/(self.sigmaW**2)
        return loss
    
    
    def Varifold_Regulocal(self):
        """
        The default dataloss : Varifold. 
        
        Returns
        -------
        @output : loss : the data attachment function.
        """
        
        if self.oriented:
            K = OrientedGaussLinKernel(sigma=self.sigmaW)
        else:
            K = GaussLinKernel(sigma=self.sigmaW)
            
        CS, LS, NSn = Compute_structures(self.VS, self.FS_sel, self.structure)
        omega_S = K(CS, CS, NSn, NSn, LS)
        CT, LT, NTn = Compute_structures(self.VT, self.FT_sel, self.structure)
    
        cst = (LT * K(CT, CT, NTn, NTn, LT)).sum()
        
        def loss(VS):
            CS1, LS1, NSn1 = Compute_structures(VS, self.FS_sel, self.structure)
            omega_S1 = K(CS1, CS1, NSn1, NSn1, LS1)
            weight = LS1/LS
            Regul_local = ((omega_S - weight*omega_S1)**2).sum()
            cost = 10*np.pi**2/4*(cst + (LS1 * omega_S1).sum() - 2 * (LS1 * K(CS1, CT, NSn1, NTn, LT)).sum())
            return (Regul_local + cost)/(self.sigmaW**2)
        return loss
    
    
    def PartialVarifold(self):
        """
        The Partial Varifold, in its global version. Designed to include the 
        deformed source into the target. 
        
        Returns
        -------
        @output : loss : the data attachment function.
        """
        if self.oriented:
            K = OrientedGaussLinKernel(sigma=self.sigmaW)
        else:
            K = GaussLinKernel(sigma=self.sigmaW)
            
        CT, LT, NTn = Compute_structures(self.VT, self.FT_sel, self.structure)
        
        def loss(VS):
            CS, LS, NSn = Compute_structures(VS, self.FS_sel, self.structure)
            cross = (LS * K(CS, CT, NSn, NTn, LT)).sum()
    
            cost_S = (LS * K(CS, CS, NSn, NSn, LS)).sum() - cross
            return cost_S*cost_S/(self.sigmaW**2) 
        return loss
    
    
    def PartialVarifoldLocal(self):
        """
        The Partial Varifold, in its local version. Designed to include the 
        deformed source into the target. 
        
        Returns
        -------
        @output : loss : the data attachment function.
        """
        def g2(x):
            return (x**2)*(x>0).float()
    
        if self.oriented:
            K = OrientedGaussLinKernel(sigma=self.sigmaW)
        else:
            K = GaussLinKernel(sigma=self.sigmaW)

        CT, LT, NTn = Compute_structures(self.VT, self.FT_sel, self.structure)
    
        def loss(VS):
            CS, LS, NSn = Compute_structures(VS, self.FS_sel, self.structure)
            xs = K(CS, CT, NSn, NTn, LT)
            cost = ( LS * g2( K(CS, CS, NSn, NSn, LS) - xs )).sum() 
            return cost/(self.sigmaW**2) 
        return loss


    def PartialVarifoldLocalNormalized(self):
        """
        The Partial Weighted Varifold, in its local version. Designed to include
        the deformed source into the target. 
        
        Returns
        -------
        @output : loss : the data attachment function.
        """

        def g2(x):
            return (x**2)*(x>0).float()

        if self.oriented:
            K = OrientedGaussLinKernel(sigma=self.sigmaW)
            WeightedKernel = PartialWeightedGaussLinKernelOriented(sigma=self.sigmaW)
        else:
            K = GaussLinKernel(sigma=self.sigmaW)
            WeightedKernel = PartialWeightedGaussLinKernel(sigma=self.sigmaW)
        
        CT, LT, NTn = Compute_structures(self.VT, self.FT_sel, self.structure)
        omega_T = K(CT, CT, NTn, NTn, LT)

        def loss(VS):
            CS, LS, NSn = Compute_structures(VS, self.FS_sel, self.structure)

            omega_S    = K(CS, CS, NSn, NSn, LS)
            omega_tild = WeightedKernel(CS, CT, NSn, NTn, omega_S, omega_T, LT)
            
            cost = ( LS * g2( omega_S - omega_tild  )).sum()

            return cost/(self.sigmaW**2) #curve (sigmaW**2)
        return loss    


    def PartialVarifoldNormalized_Regulocal(self):

        if self.oriented:
            K = OrientedGaussLinKernel(sigma=self.sigmaW)
        else:
            K = GaussLinKernel(sigma=self.sigmaW)
        
        CS, LS, NSn = Compute_structures(self.VS, self.FS_sel, self.structure)
        omega_S = K(CS, CS, NSn, NSn, LS)
        
        L2 = self.PartialVarifoldLocalNormalizedOriented()
        
        def loss(VS):
            CS1, LS1, NSn1 = Compute_structures(VS, self.FS_sel, self.structure)
            omega_S1 = K(CS1, CS1, NSn1, NSn1, LS1)
            weight = LS1/LS
            Regul_local = ((omega_S - weight*omega_S1)**2).sum()
            return Regul_local + L2(VS)
        return loss



##################################################
#### The Normal Cycle Losses -- for curves -- ####
##################################################

    def loss_NC_constant(self):
        """
        The Normal Cycle dataloss with a constant kernel.
        
        Returns
        -------
        @output : loss : the data attachment function.
        """
        K = GenericGaussKernel(self.sigmaW)
    
        NT = Compute_sum_normalized_edges_at_vertex(self.VT,self.FT_sel)
    
        cst = (NT*K(self.VT,self.VT,NT)).sum()
        
        def loss(VS):
            NS = Compute_sum_normalized_edges_at_vertex(VS,self.FS_sel)
            return 10*np.pi**2/4*(cst + ((NS*K(VS,VS,NS)).sum() - 2*(NS*K(VS,self.VT,NT)).sum())) 
        
        return loss


    def loss_NC_linear(self):
        """
        The Normal Cycle dataloss with a linear kernel.
        Designed to include the deformed source into the target. 
        
        Returns
        -------
        @output : loss : the data attachment function.
        """
    
        if self.structure == "surfaces":
            print("Normal cycles loss is not available for surfaces, to do. Returning 1.")
            return 1
    
        K = GenericGaussKernel(self.sigmaW)
        Kcyl = GaussLinKernel(self.sigmaW)
        
        centers_T, lengths_T, normalized_seg_T = Compute_structures(self.VT,self.FT_sel, self.structure)
        neighb_T = Count_Vertex_Neighb(self.FT_sel)
    
        cst_cyl = (lengths_T*Kcyl(centers_T,centers_T, normalized_seg_T, normalized_seg_T, lengths_T)).sum()
        
        ind_sel_VT   = torch.cuda.LongTensor(neighb_T.nonzero()[:,0])
        neighb_T_sel = neighb_T.index_select(0,ind_sel_VT)
        VT_sel = self.VT.index_select(0,ind_sel_VT)
        cst_sph = ((1-neighb_T_sel/2)*K(VT_sel,VT_sel,1-neighb_T_sel/2)).sum()
        
        neighb_S = Count_Vertex_Neighb(self.FS_sel)
        
        ind_sel_VS   = torch.cuda.LongTensor(neighb_S.nonzero()[:,0])
        neighb_S_sel = neighb_S.index_select(0,ind_sel_VS)
    
        def loss(VS):
            
            if(self.selected_source):
                VS_sel = VS.index_select(0,ind_sel_VS)
            else:
                VS_sel = VS
    
            cost = 0
            centers_S, lengths_S, normalized_seg_S = Compute_structures(VS,self.FS_sel, self.structure)
               
            source_cyl = (lengths_S*Kcyl(centers_S,centers_S,
                                           normalized_seg_S,normalized_seg_S,
                                           lengths_S)).sum()
            
            source_sph = ((1-neighb_S_sel/2)*K(VS_sel,VS_sel,1-neighb_S_sel/2)).sum()
        
            crossed_cyl = (lengths_S*Kcyl(centers_S,centers_T,
                                       normalized_seg_S,
                                       normalized_seg_T,
                                       lengths_T)).sum()
        
            crossed_sph = ((1-neighb_S_sel/2)*K(VS_sel,VT_sel,1-neighb_T_sel/2)).sum()
        
            cost_cyl = cst_cyl+source_cyl-2*crossed_cyl

            cost = np.pi**2/2*(cst_cyl+source_cyl-2*crossed_cyl)+16*np.pi**2/3*(cst_sph+source_sph-2*crossed_sph) #the cylindrical part is sensitive to the scale.
        
            return cost
            
        return loss


