import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy import integrate
from scipy import interpolate
import time

#Generate the grid and define the parameters
nx=101; ny=nx;
h=1/(nx-1);
W=2*h;
bp=0.5;
b=W*bp; 
W2_inv=1/W**2;
xm=np.arange(-0.1,1.1+h,h); ym=np.arange(-0.1,1.1+h,h);
[X,Y]=np.meshgrid(xm,ym);
x=np.arange(-0.1,1.1+h,h); y=(1+np.cos(2*np.pi*(1-x)))/4;
C=np.vstack((x,y)).T;   # contour
Nx=len(xm); Ny=len(ym);
Mark = np.zeros(X.shape)
for i in range(0,len(xm)):
    for j in range(0,len(ym)):
        if (Y[i,j])<(((1+np.cos(2*np.pi*(1-X[i,j])))/4)):
            Mark[i,j]=-1;
for i in range(0,len(xm)):
    for j in range(0,len(ym)):
        phi[i,j]=distance_function(X[i,j],Y[i,j],C);
        if Mark[i,j]==-1:
            phi[i,j] = -phi[i,j];
for i in range(0,len(xm)):
    for j in range(0,len(ym)):
        if phi[i,j]>0.03:
            phi[i,j]=1;
        elif phi[i,j]<-.03:
            phi[i,j]=-1;

kappa= curvature(phi,h); normphi=normcd(phi,h); d2phi=laplacian(phi,h);

plt.contour(X,Y,phi,[0,0],'b',linewidth=1);
plt.hold(True)

dt=0.001; tf=0.1;
t0=np.arange(0,0.3+0.1,0.1); tf=np.arange(0.1,0.4+0.1,0.1);
tic=time.time()
for p in range(0,len(tf)):
    for t in np.arange(t0[p],tf[p]+dt,dt):
        kappa=curvature(phi,h)
        normphi=normcd(phi,h)
        d2phi=laplacian(phi,h)
        for i in range(1,Nx):
            for j in range(1,Ny):
                phi[i,j]=phi[i,j]+dt*(b*(d2phi[i,j]+W2_inv*phi[i,j]*(1-phi[i,j]**2)-normphi[i,j]*kappa[i,j])-normphi[i,j])