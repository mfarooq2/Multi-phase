def laplacian(phi, h):
    # Calculates the lapacian of the function phi(x,y) using a 9-point 
    # finite-difference stencil
    # where h=dx=dy is the spatial grid size
    # 
    # Maged Ismail 04/25/07

    import numpy as np
    from numpy import zeros, shape

    d2phi=zeros(shape(phi))
    invh2=1/h**2
    Nx,Ny=shape(phi)

    phibc=zeros((Nx+2,Ny+2))
    phibc[1:Nx+1,1:Ny+1] = phi # Copy phi into phibc
    phibc[   0,1:Ny+1]   = phi[Nx-1,:] # Periodic bc
    phibc[Nx+1,1:Ny+1]   = phi[   0,:] # Periodic bc
    phibc[1:Nx+1,   0]   = -1 # Dirichlet  bc
    phibc[1:Nx+1,Ny+1]   =  1 # Dirichlet  bc

    phi=phibc
    Nx,Ny=shape(phi)

    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            d2phi[i,j]=2*(phi[i+1,j]+phi[i,j+1]+phi[i-1,j]+phi[i,j-1]-4*phi[i,j])\
                +0.5*(phi[i+1,j+1]+phi[i+1,j-1]+phi[i-1,j+1]+phi[i-1,j-1]-4*phi[i,j])

    d2phi=(1/3)*invh2*d2phi; d2phi=d2phi[1:-1,1:-1]
    return d2phi