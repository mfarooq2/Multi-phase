import math
import numpy as np

def GenPointer(nx, ny):
    ## Memory allocation
    ip = np.nan*np.ones((nx,ny))
    iu = np.nan*np.ones((nx,ny))

    ## Pointer matrix for P
    id_p = 0 # index to be used in vector variable P
    for i in range(0,nx):
        for j in range(0,ny):
            ip[i, j] = id_p
            id_p = id_p + 1 
    ## Pointer matrix for ux
    id_u = 0  # index to be used in vector variable u = [ux; uy]
    for i in range(1,nx):
        for j in range(0,ny):
            iu[i, j] = id_u
            id_u = id_u + 1

    ip[np.isnan(ip)]=0
    iu[np.isnan(iu)]=0
    return ip.astype(int),iu.astype(int)





def Adv_x_n_new(qi,var_dict):
        ## advection operator (BC embedded): -\nabla \cdot (uu) 
    ##      input: u-type (nu elements)
    ##       output: u-type (nu elements)
    nx1=var_dict['nx1']
    nx2=var_dict['nx2']
    np_ = var_dict['np_']
    nu = var_dict['nu']
    iu=var_dict['iu']
    ip=var_dict['ip']
    h=var_dict['h']
    #ubCT_=ubCT_function(iu,dx,nx,nu)
    ## Initialize output
    qo = np.nan*np.ones((nu,1))
    qi=qi.reshape(nu,1)
    if np.sum(np.isnan(qi))>0:
        qi=np.zeros((len(qi),1))
    ## 1. U-Component
    ## inner domain
    i=np.nan
    j=np.nan
    # UNx1=var_dict['UNx1']
    # UNx2=var_dict['UNx2']
    # qo[iu[2:nx1-1, 1:nx2-1]] = (1/h) * ( - ( qi[iu[1:nx1-2,1:nx2-1]] + qi[iu[2:nx1-1,1:nx2-1]] ) / 2 * ( qi[iu[1:nx1-2,1:nx2-1  ]] + qi[iu[2:nx1-1,1:nx2-1  ]] ) / 2   \
    #                                         + ( qi[iu[2:nx1-1,1:nx2-1]] + qi[iu[3:nx1,1:nx2-1  ]] ) / 2 * ( qi[iu[2:nx1-1,1:nx2-1  ]] + qi[iu[3:nx1,1:nx2-1  ]] ) / 2 ) 

    # advxn=np.zeros_like(un)
    
    # advxn[1:-1,1:-1]
    qo[iu[1:-1,1:-1]] = (1 / h) * (
        (0.5 * (qo[iu[1 + 1 :, 1:-1]] + qo[iu[1:-1, 1:-1]])) ** 2
        - (0.5 * (qo[iu[1:-1, 1:-1]] + qo[iu[1 - 1 : -2, 1:-1]])) ** 2
    )
    return qo

def Dif_x_n_new(qi,var_dict):
    ## Laplace operator: 
    #       input: u-type (nu elements)
    #       output: u-type (nu elements)

    ## Initialize output
    nx1=var_dict['nx1']
    nx2=var_dict['nx2']
    np_ = var_dict['np_']
    nu = var_dict['nu']
    iu=var_dict['iu']
    ip=var_dict['ip']
    h=var_dict['h']
    #ubCT_=ubCT_function(iu,dx,nx,nu)
    ## Initialize output
    qo=np.nan*np.ones((nu,1))
    if np.sum(np.isnan(qi))>0:
        qi=np.zeros((len(qi),1))

    ## 1. ex-Component
    ## inner domain
    i=np.nan
    j=np.nan

    #qo[iu[2:nx-1, 1:ny-1]] = (+qi[iu[1:nx-2, 1:ny-1]] -2*qi[iu[2:nx-1, 1:ny-1]] + qi[iu[3:nx, 1:ny-1]] ) / (dx**2) + ( +qi[iu[2:nx-1, 0:ny-2]] -2*qi[iu[2:nx-1, 1:ny-1]] + qi[iu[2:nx-1, 2:ny]] ) / (dy**2)
    
    #difxn=np.zeros([UNx1,UNx2])
    #difxn[1:-1,1:-1]
    qo[iu[1:-1,1:-1]]=(1/(h**2))*(qi[iu[1+1:,1:-1]]+qi[iu[0:-2,1:-1]]+qi[iu[1:-1,1+1:]]+qi[iu[1:-1,0:-2]]-4*qi[iu[1:-1,1:-1]])    
    return qo