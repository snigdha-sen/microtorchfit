import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as utils

__all__ = [
    'ball_stick',
    't2_adc',
    'msdki',
    'ball',
    'stick',
    'get_model_nparams']

def ball_stick(grad,params):
    
    #nparams = len(params)
    #need to force this shape?
    #params = torch.zeros([1,nparams])
     
    
    # extract the parameters
    f = params[:,0].unsqueeze(1)
    Dpar = params[:, 1].unsqueeze(1)
    D = params[:, 2].unsqueeze(1)
    theta = params[:, 3].unsqueeze(1)
    phi = params[:, 4].unsqueeze(1)


    g = grad[:,0:2]
    bvals = grad[:,3]  
    
    print(stick(grad, Dpar, theta,phi).size())
    print(ball(grad, D).size())
    
    S = f * stick(grad, Dpar, theta,phi) + (1 - f) * ball(grad, D)

    return S

def t2_adc(grad,params):
    # extract the parameters
    T2 = params[:,0].unsqueeze(1)
    D = params[:, 1].unsqueeze(1)   
    
    bvals = grad[:,3]
    te = grad[:,4]
    S = torch.exp(-grad[:,3]*D) * torch.exp(-(te - torch.min(te))/T2) 
    
    return S

def msdki(grad,params):
    
    #D = torch.clamp(params[:, 0].unsqueeze(1), min = 0.01, max = 5)
    #K = torch.clamp(params[:, 1].unsqueeze(1), min= 0.001, max=3)
    D = params[:, 0].unsqueeze(1)
    K = params[:, 1].unsqueeze(1)
    
    bvals = grad[:,3]
        
    S = torch.exp(-bvals*D + (bvals**2 * D**2 * K / 6)) 
    
    return S



def ball(grad, D):
    bvals = grad[:, 3].unsqueeze(1)

    S = torch.exp(-bvals * D)
    return S


def stick(grad, Dpar, theta, phi):
    
    g = grad[:, 0:3]
    bvals = grad[:, 3].unsqueeze(1)

    n = sphere2cart(theta,phi)
    
    S = torch.exp(-bvals * Dpar * torch.mm(g, n) ** 2)
    
    return S


def sphere2cart(theta,phi):   
    n = torch.zeros(3,theta.size(0))
    
    sintheta = torch.sin(theta)
    
    n[0,:] = torch.squeeze(sintheta * torch.cos(phi))
    n[1,:] = torch.squeeze(sintheta * torch.sin(phi))
    n[2,:] = torch.squeeze(torch.cos(theta))   
    
    return n
    
    
def cart2sphere(xyz):
    shape = xyz.shape[:-1]
    mu = np.zeros(np.r_[shape, 2])
    r = np.linalg.norm(xyz, axis=-1)
    mu[..., 0] = np.arccos(xyz[..., 2] / r)  # theta
    mu[..., 1] = np.arctan2(xyz[..., 1], xyz[..., 0])
    mu[r == 0] = 0, 0
    return mu



def get_model_nparams(model):
    if model=="ball_stick":
        return 5
    if model=="t2_adc":
        return 2
    if model=="msdki":
        return 2




