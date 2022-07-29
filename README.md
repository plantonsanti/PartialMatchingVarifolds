# PartialMatchingVarifolds
This is the code associated to the paper *Partial Matching in the Space of Varifolds* from *The 27th international conference on Information Processing in Medical Imaging (June, 2021)* (preprint : https://arxiv.org/abs/2103.12441 )

<pre>

Dependancies : 
- numpy
- torch
- pykeops (see : https://www.kernel-operations.io/keops/python/installation.html)

The examples can be run with the scripts :<br/>
example_surfaces.py 
example_vascular_trees.py 

Each example save the results in /results/surfaces 
                                         /vascular_trees 
                                         
With the files :  
<strong>scale_N.npz</strong>             : the output mesh (tree or surface) of the deformation at the different data attachment scales <strong>N</strong>. 
<strong>Momenta_scale_N.npy</strong>     : the initial momenta to use to reproduce the deformation at the different data attachment scales <strong>N</strong>. 
<strong>resume_optimization.txt</strong> : the number of iterations and the loss function value at each data attachment scale <strong>N</strong>.  

<strong>control_points.npy</strong>      : control points for the final deformation.  
<strong>momenta2apply.npy</strong>       : the initial momenta to use to reproduce final deformation. 



The optimization of the loss function at each data attachment scale is monitored and saved in /results/surfaces/dict_resume_opt/ 
</pre>
