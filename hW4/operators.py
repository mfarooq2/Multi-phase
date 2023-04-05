import numpy as np
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

def Laplace_Vec(qi,uBC_L, uBC_R, uBC_B, uBC_T, vBC_L, vBC_R, vBC_T, vBC_B,np_,nu,nx,ny,dx,dy,iu,iv,ip,dt,v,idu):
    
    ## Laplace operator: 
    #       input: u-type (nu elements)
    #       output: u-type (nu elements)

    ## Initialize output
    qo=np.nan*np.ones((nu,1))
    if np.sum(np.isnan(qi))>0:
        qi=np.zeros((len(qi),1))

    ## 1. ex-Component
    ## inner domain
    i=np.nan
    j=np.nan

    qo[iu[2:nx-1, 1:ny-1]] = (+qi[iu[1:nx-2, 1:ny-1]] -2*qi[iu[2:nx-1, 1:ny-1]] + qi[iu[3:nx, 1:ny-1]] ) / (dx**2) + ( +qi[iu[2:nx-1, 0:ny-2]] -2*qi[iu[2:nx-1, 1:ny-1]] + qi[iu[2:nx-1, 2:ny]] ) / (dy**2)

    #qo[iu[i, j]] = (+qi[iu[i-1, j]] -2*qi[iu[i, j]] + qi[iu[i+1, j]] ) / (dx**2) + ( +qi[iu[i, j-1]] -2*qi[iu[i, j]] + qi[iu[i, j+1]] ) / (dy**2) qo[iu[2:nx-1, 1:ny-1]]

    ## Edges
    # left inner 
    i = 1
    j=np.nan
    qo[iu[i, 1:ny-1]] = (-2*qi[iu[i, 1:ny-1]] + qi[iu[i+1, 1:ny-1]] ) / (dx**2) + ( +qi[iu[i, 0:ny-2]] -2*qi[iu[i, 1:ny-1]] + qi[iu[i, 2:ny]] ) / (dy**2) # + uBC_L / (dx^2)

    # bottom inner
    j = 0
    i=np.nan
    qo[iu[2:nx-1, j]] =   ( +qi[iu[1:nx-2, j]] -2*qi[iu[2:nx-1, j]] + qi[iu[3:nx, j]] ) / (dx**2) + ( -qi[iu[2:nx-1, j]]   -2*qi[iu[2:nx-1, j]] + qi[iu[2:nx-1, j+1]] ) / (dy**2) #   ## + 2*uBC_B / (dy^2) 

    #right inner
    i=-1
    j=np.nan
    qo[iu[i, 1:ny-1]] = (+qi[iu[i-1, 1:ny-1]] -2*qi[iu[i, 1:ny-1]] ) / (dx**2) + ( +qi[iu[i, 0:ny-2]] -2*qi[iu[i, 1:ny-1]] + qi[iu[i, 2:ny]] ) / (dy**2) # qi[iu[i+1, j]] 
    

    #top inner
    j=-1
    i=np.nan
    qo[iu[2:nx-1, j]] = (+qi[iu[1:nx-2, j]] -2*qi[iu[2:nx-1, j]] + qi[iu[3:nx, j]] ) / (dx**2) + ( -qi[iu[2:nx-1, j]]+qi[iu[2:nx-1, j-1]] -2*qi[iu[2:nx-1, j]]  ) / (dy**2)  #+ qi[iu[i, j+1]]
    

    ## Corners
    # bottom left
    i = 1 
    j = 0
    qo[iu[i, j]] =(-2*qi[iu[i, j]] + qi[iu[i+1, j]] ) / (dx**2) + ( -qi[iu[i, j]]   -2*qi[iu[i, j]] + qi[iu[i, j+1]] ) / (dy**2) # + uBC_L   / (dx^2)   # + 2*uBC_B / (dy^2) 

    # bottom right
    i=-1
    j=0    
    qo[iu[i, j]] = (+qi[iu[i-1, j]] -2*qi[iu[i, j]]  ) / (dx**2) + (-qi[iu[i, j]]-2*qi[iu[i, j]] + qi[iu[i, j+1]] ) / (dy**2)  #+ qi[iu[i+1, j]] +qi[iu[i, j-1]]
  
    # top left
    i=1
    j=-1
    qo[iu[i, j]] = ( -2*qi[iu[i, j]] + qi[iu[i+1, j]] ) / (dx**2) + (-qi[iu[i, j]] +qi[iu[i, j-1]] -2*qi[iu[i, j]]  ) / (dy**2)  ##+qi[iu[i-1, j]] + qi[iu[i, j+1]]
    

    # top right
    i=-1
    j=-1
    qo[iu[i, j]] = (+qi[iu[i-1, j]] -2*qi[iu[i, j]]  ) / (dx**2) + ( -qi[iu[i, j]]+qi[iu[i, j-1]] -2*qi[iu[i, j]]  ) / (dy**2) #+ qi[iu[i+1, j]] + qi[iu[i, j+1]]
