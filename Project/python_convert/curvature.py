import numpy as np

def curvature(phi,h):  # sourcery skip: low-code-quality

    invhh=1/h
    Nx,Ny=phi.shape
    
    phibc = np.zeros((Nx+1,Ny+1))
    phibc[1:Nx+1,1:Ny+1]=phi
    phibc[0,1:Ny+1]=phi[Nx-1,:]
    phibc[Nx,1:Ny+1]=phi[0,:]
    phibc[1:Nx+1,0]=-1.0
    phibc[1:Nx+1,Ny]= 1.0
    phi=phibc
    
    phi=phibc; Nx,Ny=phi.shape
    term1=np.zeros(phi.shape);term2=term1; term3=term1; term4=term1;
    denom1=np.zeros(phi.shape); denom2=denom1; denom3=denom1; denom4=denom1;

    for i in range(1,Nx-1):
        for j in range(1,Ny-1):
            denom1[i,j]=np.sqrt((phi[i+1,j]-phi[i,j])**2+(1/16)*\
                (phi[i+1,j+1]+phi[i,j+1]-phi[i+1,j-1]-phi[i,j-1])**2);
            denom2[i,j]=np.sqrt((phi[i,j]-phi[i-1,j])**2+(1/16)*\
                (phi[i-1,j+1]+phi[i,j+1]-phi[i-1,j-1]-phi[i,j-1])**2);      
            denom3[i,j]=np.sqrt((phi[i,j+1]-phi[i,j])**2+(1/16)*\
                (phi[i+1,j+1]+phi[i+1,j]-phi[i-1,j+1]-phi[i-1,j])**2);
            denom4[i,j]=np.sqrt((phi[i,j]-phi[i,j-1])**2+(1/16)*(phi[i+1,j-1]+phi[i+1,j]-phi[i-1,j-1]-phi[i-1,j])**2)
            if denom1[i,j]==0:
                term1[i,j]=0
            else:
                term1[i,j]=(phi[i+1,j]-phi[i,j])/denom1[i,j]
            if denom2[i,j]==0:
                term2[i,j]=0
            else:
                term2[i,j]=(phi[i-1,j]-phi[i,j])/denom2[i,j]
            if denom3[i,j]==0:
                term3[i,j]=0
            else:
                term3[i,j]=(phi[i,j+1]-phi[i,j])/denom3[i,j]
            if denom4[i,j]==0:
                term4[i,j]=0
            else:
                term4[i,j]=(phi[i,j-1]-phi[i,j])/denom4[i,j]
    kappa=invhh*(term1+term2+term3+term4)
    #kappa=kappa[1:-2,1:-2]
    
    return kappa