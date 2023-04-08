import itertools
import numpy as np
import math
#import matplotlib
import matplotlib.pyplot as plt
#import numba as nb
import threading as thd
#import logging as lg
from util_tools.operators import *
from util_tools.update_funcs import *
#problem constants
var_dict = {}
var_dict['nu']=1e-6
var_dict['mu']=1e-3
var_dict['rho']=1e+3
var_dict['st_coef']=0.06
#real timestep
var_dict['dt']=0.0001
var_dict['gradP']=-2.4

n_iter=0
global epstot
'''
node generation section
'''
#domain length

Lx1=var_dict['Lx1']=0.02
Lx2=var_dict['Lx2']=0.01

r_dpl=var_dict['r_dpl']=0.25*var_dict['Lx2']
#number of cells on each direction
Nx1=var_dict['Nx1']=120
Nx2=var_dict['Nx2']=60

cell_vol=var_dict['cell_vol']=(Lx1/Nx1)*(Lx2/Nx2)

#mesh spacing
h=var_dict['h']=Lx1/Nx1

#redistancing pseudo-time count
tau=0.0
#redistancing pseudo-time step
dtau=0.5*h
#smoothing range
M=var_dict['M']=3.0
#uave
u1_ave=var_dict['u1_ave']=0.02

    
epstot=100.0
p_iter=0
#cell centroid coor
#the +2 stands for ghost cells on each direction
cell_cent_x=np.zeros([Nx1+2,Nx2+2])
cell_cent_y=np.zeros([Nx1+2,Nx2+2])

cell_cent_pn=np.zeros([Nx1+2,Nx2+2])
cell_cent_pnn=np.zeros([Nx1+2,Nx2+2])

cell_cent_phin=np.zeros([Nx1+2,Nx2+2])
cell_cent_phis=np.zeros([Nx1+2,Nx2+2])
cell_cent_phinn=np.zeros([Nx1+2,Nx2+2])

cell_cent_phi_dn=np.zeros([Nx1+2,Nx2+2])
cell_cent_phi_ds=np.zeros([Nx1+2,Nx2+2])
cell_cent_phi_dnn=np.zeros([Nx1+2,Nx2+2])

cell_cent_rho=np.zeros([Nx1+2,Nx2+2])
cell_cent_mu=np.zeros([Nx1+2,Nx2+2])
#cell corner coor
cell_cor_x=np.zeros([Nx1+3,Nx2+3])
cell_cor_y=np.zeros([Nx1+3,Nx2+3])

#surf area of the cell 
cell_S_x=np.zeros([Nx1+2,Nx2+2])
cell_S_y=np.zeros([Nx1+2,Nx2+2])

cell_S_x_coor_x=np.zeros([Nx1+2,Nx2+2])
cell_S_x_coor_y=np.zeros([Nx1+2,Nx2+2])
cell_S_y_coor_x=np.zeros([Nx1+2,Nx2+2])
cell_S_y_coor_y=np.zeros([Nx1+2,Nx2+2])

#normal vector of cell surfaces
cell_S_x_nx=np.zeros([Nx1+2,Nx2+2])
cell_S_x_ny=np.zeros([Nx1+2,Nx2+2])
cell_S_y_nx=np.zeros([Nx1+2,Nx2+2])
cell_S_y_ny=np.zeros([Nx1+2,Nx2+2])
#surface velocities
cell_S_x_un=np.zeros([Nx1+2,Nx2+2])
cell_S_x_us=np.zeros([Nx1+2,Nx2+2])
cell_S_x_unn=np.zeros([Nx1+2,Nx2+2])

cell_S_y_vn=np.zeros([Nx1+2,Nx2+2])
cell_S_y_vs=np.zeros([Nx1+2,Nx2+2])
cell_S_y_vnn=np.zeros([Nx1+2,Nx2+2])
#reference velocity profile
ref_S_u=np.zeros([Nx2+2])
L_sq=np.array([1.0,1.0])

#corner coor initialization
cell_cor_x, cell_cor_y = np.meshgrid(np.linspace(-Lx1/2, Lx1/2, Nx1+3), np.linspace(-Lx2/2, Lx2/2, Nx2+3))
        
#cell cent coor storage
        
cell_cent_x[0:Nx1+2,0:Nx2+2]='{:10.6e}'.format(0.25*(cell_cor_x[0:Nx1+2,0:Nx2+2]+cell_cor_x[0+1:Nx1+2+1,0:Nx2+2]+cell_cor_x[0:Nx1+2,0+1:Nx2+2+1]+cell_cor_x[0+1:Nx1+2+1,0+1:Nx2+2+1]))
cell_cent_y[0:Nx1+2,0:Nx2+2]='{:10.6e}'.format(0.25*(cell_cor_y[0:Nx1+2,0:Nx2+2]+cell_cor_y[0+1:Nx1+2+1,0:Nx2+2]+cell_cor_y[0:Nx1+2,0+1:Nx2+2+1]+cell_cor_y[0+1:Nx1+2+1,0+1:Nx2+2+1]))
#lvlset init
cell_cent_phin[0:Nx1+2,0:Nx2+2]=lvlset_init(cell_cent_x[0:Nx1+2,0:Nx2+2], cell_cent_y[0:Nx1+2,0:Nx2+2],var_dict)
cell_cent_rho[0:Nx1+2,0:Nx2+2]=rho_distr(i,j)
cell_cent_mu[0:Nx1+2,0:Nx2+2]=mu_distr(i,j)
cell_S_x_coor_x[0:Nx1+2,0:Nx2+2]=(cell_cor_x[0:Nx1+2,0:Nx2+2]+cell_cor_x[0:Nx1+2,0+1:Nx2+2+1])/2
cell_S_x_coor_y[0:Nx1+2,0:Nx2+2]=(cell_cor_y[0:Nx1+2,0:Nx2+2]+cell_cor_y[0:Nx1+2,0+1:Nx2+2+1])/2
cell_S_y_coor_x[0:Nx1+2,0:Nx2+2]=(cell_cor_x[0:Nx1+2,0:Nx2+2]+cell_cor_x[0+1:Nx1+2+1,0:Nx2+2])/2
cell_S_y_coor_y[0:Nx1+2,0:Nx2+2]=(cell_cor_y[0:Nx1+2,0:Nx2+2]+cell_cor_y[0+1:Nx1+2+1,0:Nx2+2])/2
#initial conditions
cell_S_x_un[0:Nx1+2,0:Nx2+2]=ref_vel_prof(cell_cent_y[0:Nx1+2,0:Nx2+2])
#cell_S_y_un[i,j]=0.00

cell_S_x[0:Nx1+2,0:Nx2+2]=abs(cell_cor_y[0:Nx1+2,0:Nx2+2]-cell_cor_y[0:Nx1+2,0+1:Nx2+2+1])
cell_S_y[0:Nx1+2,0:Nx2+2]=abs(cell_cor_x[0:Nx1+2,0:Nx2+2]-cell_cor_x[0+1:Nx1+2+1,0:Nx2+2])

cell_S_x_nx[0:Nx1+2,0:Nx2+2]=(cell_cor_y[0:Nx1+2,0+1:Nx2+2+1]-cell_cor_y[0:Nx1+2,0:Nx2+2])/cell_S_x[0:Nx1+2,0:Nx2+2]
cell_S_x_ny[0:Nx1+2,0:Nx2+2]=(cell_cor_x[0:Nx1+2,0+1:Nx2+2+1]-cell_cor_x[0:Nx1+2,0:Nx2+2])/cell_S_x[0:Nx1+2,0:Nx2+2]
cell_S_y_nx[0:Nx1+2,0:Nx2+2]=(cell_cor_y[0+1:Nx1+2+1,0:Nx2+2]-cell_cor_y[0:Nx1+2,0:Nx2+2])/cell_S_y[0:Nx1+2,0:Nx2+2]
cell_S_y_ny[0:Nx1+2,0:Nx2+2]=(cell_cor_x[0+1:Nx1+2+1,0:Nx2+2]-cell_cor_x[0:Nx1+2,0:Nx2+2])/cell_S_y[0:Nx1+2,0:Nx2+2]        

bub=plt.Circle((0.01, 0.005), r_dpl, color='grey', fill=False)
fig, ax=plt.subplots()
plt.contourf(cell_cent_x[1:Nx1+1, 1:Nx2+1], cell_cent_y[1:Nx1+1, 1:Nx2+1], cell_cent_phin[1:Nx1+1, 1:Nx2+1], 20, cmap='coolwarm')
plt.colorbar()
ax.add_artist(bub)
plt.xlabel('$x_1$ (m)')
plt.ylabel('$x_2$ (m)')
plt.title('domain initial level-set, '+str(Nx2), fontsize=9)
plt.gca().set_aspect('equal')
plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_ls_init.png')
plt.show() 

plt.contourf(cell_cent_x[1:Nx1+1, 1:Nx2+1], cell_cent_y[1:Nx1+1, 1:Nx2+1], cell_cent_rho[1:Nx1+1, 1:Nx2+1], 20, cmap='cool')
plt.colorbar()
ax.add_artist(bub)
plt.xlabel('$x_1$ (m)')
plt.ylabel('$x_2$ (m)')
plt.title('domain initial rho $(kg/m^3)$, '+str(Nx2), fontsize=9)
plt.gca().set_aspect('equal')
plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_rho_init.png')
plt.show() 

plt.contourf(cell_cent_x[1:Nx1+1, 1:Nx2+1], cell_cent_y[1:Nx1+1, 1:Nx2+1], cell_cent_mu[1:Nx1+1, 1:Nx2+1], 20, cmap='bone')
plt.colorbar()
ax.add_artist(bub)
plt.xlabel('$x_1$ (m)')
plt.ylabel('$x_2$ (m)')
plt.title('domain initial viscosity $(Pa s)$, '+str(Nx2), fontsize=9)
plt.gca().set_aspect('equal')
plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_mu_init.png')
plt.show() 

L_sq_r=L_sq[1]/L_sq[0]
for i in range(1, Nx2+1):
    ref_S_u[i]=ref_vel_prof(cell_S_x_coor_y[0,i])  

while n_iter<=1300:
    L_sq[0]=L_sq[1]
    if __name__ == "__main__": 
        us_looper_master()
    
    epstot=100.0
    while epstot>1e-3:
        epstot=0.0

        if __name__ == "__main__": 
            p_looper_master()
        
    
        if p_iter%1000==0:
            plt.contourf(cell_cent_x[1:Nx1+1, 1:Nx2+1], cell_cent_y[1:Nx1+1, 1:Nx2+1], cell_cent_pnn[1:Nx1+1, 1:Nx2+1], 20, cmap='jet')
            plt.colorbar()
            plt.xlabel('$x_1$ (m)')
            plt.ylabel('$x_2$ (m)')
            plt.title('domain $p^{n+1}$ contour ($Pa$)')
            plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_p_contour.png')
            plt.gca().set_aspect('equal')
            plt.show() 
            print('eps_tot= {:5.4e}'.format(epstot))
        p_iter+=1

    #cell_S_x_unn=unn_looper(cell_S_x_us, cell_cent_pnn, cell_S_x_unn)
    if __name__ == "__main__": 
        unn_looper_master() 
        
    cell_S_x_un=cell_S_x_unn
   
    if __name__ == "__main__": 
        phis_looper_master()
    if __name__ == "__main__": 
        phinn_looper_master()
        
    cell_cent_phin=cell_cent_phinn
        
    #level-set redistancing
    
    tau=0.0
    while tau<=M*h:
        cell_cent_phi_dn=cell_cent_phin

        if __name__ == "__main__": 
            phi_ds_looper_master()
        if __name__ == "__main__":
            phi_dnn_looper_master()

        cell_cent_phin=cell_cent_phi_dn
        tau+=dtau
 
        for j in range(0, Nx2+2):
            for i in range(0, Nx1+2):
                cell_cent_rho[i,j]=rho_distr(i,j)
                cell_cent_mu[i,j]=mu_distr(i,j)
    
    sq_sum_error=0
    
    for i in range(1,Nx2+1):
        sq_sum_error+=(ref_S_u[i]-cell_S_x_un[int(0.5*Nx1),i])**2
    L_sq[1]=math.sqrt(sq_sum_error/(Nx2+1))
    #plotting group
    if n_iter%1000==0:
        print('iter= '+str(n_iter)+', L_sq= {:.4e}'.format(L_sq[0]))
        plt.plot(cell_S_x_un[int(0.5*Nx1),1:Nx2+1],cell_S_x_coor_y[int(0.5*Nx1),1:Nx2+1], color='navy', label='numerical sol, $L^2$= {:10.4e}'.format(L_sq[0]))
        plt.plot(ref_S_u[1:Nx2+1] ,cell_S_x_coor_y[int(0.5*Nx1),1:Nx2+1], color='red', label='reference')
        plt.xlabel('$u_1$ ($m/s$)')
        plt.ylabel('$x_2$ (m)')
        plt.legend()
        plt.grid()
        plt.gca().set_aspect('equal')
        plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_v_profile.png')
        plt.show()
        
        plt.contourf(cell_cent_x[1:Nx1+1, 1:Nx2+1], cell_cent_y[1:Nx1+1, 1:Nx2+1], cell_S_x_un[1:Nx1+1, 1:Nx2+1], 20, cmap='inferno')
        plt.colorbar()
        plt.xlabel('$x_1$ (m)')
        plt.ylabel('$x_2$ (m)')
        plt.title('domain $u_1$ contour ($m/s$)')
        plt.gca().set_aspect('equal')
        plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_v_contour.png')
        plt.show()
        
        plt.contourf(cell_cent_x[1:Nx1+1, 1:Nx2+1], cell_cent_y[1:Nx1+1, 1:Nx2+1], cell_cent_pn[1:Nx1+1, 1:Nx2+1], 20, cmap='jet')
        plt.colorbar()
        plt.xlabel('$x_1$ (m)')
        plt.ylabel('$x_2$ (m)')
        plt.title('domain $p^{n+1}$ contour ($m/s$)')
        plt.gca().set_aspect('equal')
        plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_p_contour.png')
        plt.show()
        
        plt.contourf(cell_cent_x[1:Nx1+1, 1:Nx2+1], cell_cent_y[1:Nx1+1, 1:Nx2+1], cell_cent_phin[1:Nx1+1, 1:Nx2+1], 20, cmap='coolwarm')
        plt.colorbar()
        plt.xlabel('$x_1$ (m)')
        plt.ylabel('$x_2$ (m)')
        plt.title('domain level-set')
        plt.gca().set_aspect('equal')
        plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_ls.png')
        plt.show()
        
        plt.contourf(cell_cent_x[1:Nx1+1, 1:Nx2+1], cell_cent_y[1:Nx1+1, 1:Nx2+1], cell_cent_rho[1:Nx1+1, 1:Nx2+1], 20, cmap='cool')
        plt.colorbar()
        ax.add_artist(bub)
        plt.xlabel('$x_1$ (m)')
        plt.ylabel('$x_2$ (m)')
        plt.title('domain initial rho $(kg/m^3)$, '+str(Nx2), fontsize=9)
        plt.gca().set_aspect('equal')
        plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_rho_contour.png')
        plt.show() 
        
        plt.contourf(cell_cent_x[1:Nx1+1, 1:Nx2+1], cell_cent_y[1:Nx1+1, 1:Nx2+1], cell_cent_mu[1:Nx1+1, 1:Nx2+1], 20, cmap='bone')
        plt.colorbar()
        ax.add_artist(bub)
        plt.xlabel('$x_1$ (m)')
        plt.ylabel('$x_2$ (m)')
        plt.title('domain initial viscosity $(Pa s)$, '+str(Nx2), fontsize=9)
        plt.gca().set_aspect('equal')
        plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_mu_contour.png')
        plt.show()  
        
        print('{:10d},  {:5.7e}'.format(n_iter, L_sq[1]))
    L_sq_r=L_sq[1]/L_sq[0]
    n_iter+=1
    

print('iter= '+str(n_iter)+', L_sq= {:.4e}'.format(L_sq[0]))
plt.plot(cell_S_x_un[int(0.5*Nx1),1:Nx2+1],cell_S_x_coor_y[int(0.5*Nx1),1:Nx2+1], color='navy', label='numerical sol, $L^2$= {:10.4e}'.format(L_sq[0]))
plt.plot(ref_S_u[1:Nx2+1] ,cell_S_x_coor_y[int(0.5*Nx1),1:Nx2+1], color='red', label='reference')
plt.xlabel('$u_1$ ($m/s$)')
plt.ylabel('$x_2$ (m)')
plt.legend()
plt.grid()
plt.gca().set_aspect('equal')
plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_v_profile.png')
plt.show()

plt.contourf(cell_cent_x[1:Nx1+1, 1:Nx2+1], cell_cent_y[1:Nx1+1, 1:Nx2+1], cell_S_x_un[1:Nx1+1, 1:Nx2+1], 20, cmap='inferno')
plt.colorbar()
plt.xlabel('$x_1$ (m)')
plt.ylabel('$x_2$ (m)')
plt.title('domain $u_1$ contour ($m/s$)')
plt.gca().set_aspect('equal')
plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_v_contour.png')
plt.show()

plt.contourf(cell_cent_x[1:Nx1+1, 1:Nx2+1], cell_cent_y[1:Nx1+1, 1:Nx2+1], cell_cent_pn[1:Nx1+1, 1:Nx2+1], 20, cmap='jet')
plt.colorbar()
plt.xlabel('$x_1$ (m)')
plt.ylabel('$x_2$ (m)')
plt.title('domain $p^{n+1}$ contour ($m/s$)')
plt.gca().set_aspect('equal')
plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_p_contour.png')
plt.show()

plt.contourf(cell_cent_x[1:Nx1+1, 1:Nx2+1], cell_cent_y[1:Nx1+1, 1:Nx2+1], cell_cent_phin[1:Nx1+1, 1:Nx2+1], 20, cmap='coolwarm')
plt.colorbar()
plt.xlabel('$x_1$ (m)')
plt.ylabel('$x_2$ (m)')
plt.title('domain level-set')
plt.gca().set_aspect('equal')
plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_ls.png')
plt.show()

plt.contourf(cell_cent_x[1:Nx1+1, 1:Nx2+1], cell_cent_y[1:Nx1+1, 1:Nx2+1], cell_cent_rho[1:Nx1+1, 1:Nx2+1], 20, cmap='cool')
plt.colorbar()
ax.add_artist(bub)
plt.xlabel('$x_1$ (m)')
plt.ylabel('$x_2$ (m)')
plt.title('domain initial rho $(kg/m^3)$, '+str(Nx2), fontsize=9)
plt.gca().set_aspect('equal')
plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_rho_contour.png')
plt.show() 

plt.contourf(cell_cent_x[1:Nx1+1, 1:Nx2+1], cell_cent_y[1:Nx1+1, 1:Nx2+1], cell_cent_mu[1:Nx1+1, 1:Nx2+1], 20, cmap='bone')
plt.colorbar()
ax.add_artist(bub)
plt.xlabel('$x_1$ (m)')
plt.ylabel('$x_2$ (m)')
plt.title('domain initial viscosity $(Pa s)$, '+str(Nx2), fontsize=9)
plt.gca().set_aspect('equal')
plt.savefig('hw6_2_'+str(Nx2)+'_init_ref_mu_contour.png')
plt.show() 