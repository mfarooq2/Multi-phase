import numpy as np
import matplotlib.pyplot as plt
from curvature_func import curvature
from normcd_func import normcd
from laplacian_func import laplacian
from evolve_vector_ENO3_func import evolve_vector_ENO3
from upwind_ENO3_func import upwind_ENO3
from drop_mover import evolve_vector
Nx=161
Ny=Nx
h=1/(Nx-1)
W=2*h
dt=0.001; tf=0.005;
bp=0.5
b=W*bp
W2_inv=1/(W**2)
xm=np.arange(0,1+h,h)
ym=np.arange(0,1+h,h)
x=np.arange(-0.1,1.1+h,h)
y=(1+np.cos(2*np.pi*(1-x)))/4
[X,Y]=np.meshgrid(xm,ym)
xc=.25; yc=.25; r=.15
phi = r - np.sqrt((X - xc )**2 + ( Y - yc )**2 )
phi=np.where(phi>0.03,1,-1)
#plt.contourf(X,Y,phi)
u_ext=1; v_ext=1; 
phi=evolve_vector(phi, h, u_ext, v_ext)