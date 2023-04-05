import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import norm
from tqdm import tqdm
def GenPointer(nx, ny):
    ## Memory allocation
    ip = np.nan*np.ones((nx,ny))
    iu = np.nan*np.ones((nx,ny))
    iv = np.nan*np.ones((nx,ny))
    idu = np.nan*np.ones((nx,ny))

    ## Pointer matrix for P
    id_p = 0 # index to be used in vector variable P
    for i in range(0,nx):
        for j in range(0,ny):
            ip[i, j] = id_p
            id_p = id_p + 1 
        ## Pointer matrix for P
    
    id_uni = 0 # index to be used for universal_calculation
    for i in range(0,nx):
        for j in range(0,ny):
            idu[i, j] = id_uni
            id_uni = id_uni + 1 

    ## Pointer matrix for ux
    id_u = 0  # index to be used in vector variable u = [ux; uy]
    for i in range(1,nx):
        for j in range(0,ny):
            iu[i, j] = id_u
            id_u = id_u + 1

    ## Pointer matrix for uy
    for i in range(0,nx):
        for j in range(1,ny):
            iv[i, j] = id_u
            id_u = id_u + 1
    ip[np.isnan(ip)]=0
    iv[np.isnan(iv)]=0
    iu[np.isnan(iu)]=0
    idu[np.isnan(idu)]=0
    return ip.astype(int),iu.astype(int),iv.astype(int),idu.astype(int)