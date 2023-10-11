import numpy as np
import torch

def load_grad(grad_filename): 
    #TO DO: replace with something that finds the file e.g. pkg_resources.resource_filename
    #grad_files_path = '/Users/paddyslator/python/self-qmri/data'

    grad = torch.tensor(np.loadtxt(grad_filename), dtype=torch.float32)  
    
    return grad


# def load_bvals_bvecs_files(bvals_filename, bvecs_filename):

# return grad
