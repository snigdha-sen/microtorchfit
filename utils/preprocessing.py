import numpy as np 

def direction_average(img,grad):
    #find unique shells - all parameters except gradient directions are the same
    unique_shells = np.unique(grad[:,3:], axis=0)
    
    #preallocate
    da_img = np.zeros(img.shape[0:3] + (unique_shells.shape[0],))
    da_grad = np.zeros((unique_shells.shape[0],grad.shape[1]))

    for shell, i in zip(unique_shells,range(0,unique_shells.shape[0])):
        #indices of grad file for this shell    
        shell_index = np.all(grad[:,3:] == shell, axis=1)
        #calculate the spherical mean of this shell - average along final axis    
        da_img[...,i] = np.mean(img[...,shell_index], axis=img.ndim-1)
        #fill in this row of the direction-averaged grad file       
        da_grad[i,3:] = shell
                               
    return da_img, da_grad
         
        

