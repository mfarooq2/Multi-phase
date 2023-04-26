import numpy as np
def normcd(phi,h):
    """Calculates the norm of the advection term |grad(phi(x,y))| using 
    a central difference scheme
    where h=dx=dy is the spatial grid size
    """
    import numpy as np
    normphi=np.zeros_like(phi)
    invhh=1.0/h
    Nx,Ny=phi.shape
    
    phibc=np.empty((Nx+1,Ny+1))
    
    phibc[1:Nx+1,1:Ny+1]=phi
    
    phibc[0,1:Ny+1]=phi[Nx-1,:]
    
    phibc[Nx,1:Ny+1]=phi[0,:]
    
    phibc[1:Nx+1,   0]=-1.0
    phibc[1:Nx+1,Ny]= 1.0
    phi=phibc
    Nx,Ny=phi.shape
    
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            normphi[i,j]=np.sqrt(((phi[i+1,j]-phi[i-1,j])**2)+((phi[i,j+1]-phi[i,j-1])**2))
    normphi=0.5*invhh*normphi
    return normphi