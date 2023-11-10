import numpy as np
import torch
import signal_models

def ModelMaker(comps): #makes a model FUNCTION
    class ModelFuncMaker:
        def __call__(self, grad, params, f):                                                                          
            if not type(comps) == tuple:
                #signal equation for one compartment model 
                S = comps(grad,params)
                        
                
            elif len(comps) == 2:
                #signal equation for two compartment model
                S = f * comps[0](grad, params) \
                    + (1-f) * comps[1](grad, params)   
                                    
            elif len(comps) == 3:
                #signal equation for three compartment model
                
                S = f[0] * comps[0](grad, params) \
                    + f[1] * comps[1](grad, params) \
                    + (1 - f[0] - f[1]) * comps[2](grad, params)                                                
            
            elif len(comps) == 4:
                #signal equation for four compartment model                   
                
                S = f[0] * comps[0](grad, params) \
                    + f[1] * comps[1](grad, params) \
                    + f[2] * comps[2](grad, params) \
                    + (1 - f[1] - f[2] - f[3]) * comps[3](grad, params)

            #return the signal for the appropriate model
            return S
        
    #make an instance of the model function
    modelfunc = ModelFuncMaker() 
    return modelfunc
    
    

        
      


            