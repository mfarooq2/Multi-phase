def laplacian(phi, h):
    # Calculates the lapacian of the function phi(x,y) using a 9-point 
    # finite-difference stencil
    # where h=dx=dy is the spatial grid size
    # 
    # Maged Ismail 04/25/07

    import numpy as np

    d2phi=np.zeros(np.shape(phi))
    invh2=1/(h**2)
    Nx,Ny=np.shape(phi)

    phibc=np.empty((Nx+1,Ny+1))
    
    phibc[1:Nx+1,1:Ny+1]=phi
    
    phibc[0,1:Ny+1]=phi[Nx-1,:]
    
    phibc[Nx,1:Ny+1]=phi[0,:]
    
    phibc[1:Nx+1,0]=-1.0
    phibc[1:Nx+1,Ny]= 1.0
    phi=phibc
    Nx,Ny=np.shape(phi)

    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            d2phi[i,j]=2*(phi[i+1,j]+phi[i,j+1]+phi[i-1,j]+phi[i,j-1]-4*phi[i,j])\
                +0.5*(phi[i+1,j+1]+phi[i+1,j-1]+phi[i-1,j+1]+phi[i-1,j-1]-4*phi[i,j])
    d2phi=(1/3)*invh2*d2phi
    return d2phi