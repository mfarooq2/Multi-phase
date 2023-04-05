import numpy as np
import math
import matplotlib.pyplot as plt
# =======================================================================================
# laminar flow analytical solution
def lam_gradp(mu, Lx2, u1_max):

    return u1_max*2*mu*(-1)*(2/Lx2)**2

def Adv_x_n(i,j):
    return (1/h)*((0.5*(cell_S_x_un[i,j]+cell_S_x_un[i+1,j]))**2-(0.5*(cell_S_x_un[i,j]+cell_S_x_un[i-1,j]))**2+(0.5*(cell_S_x_un[i,j+1]+cell_S_x_un[i,j]))*(cell_S_y_vn[i,j+1]+cell_S_y_vn[i+1,j+1])-(0.5*(cell_S_x_un[i,j]+cell_S_x_un[i,j-1]))*(0.5*(cell_S_y_vn[i,j]+cell_S_y_vn[i+1,j])))
def Dif_x_n(i,j):
    return (1/(h**2))*(cell_S_x_un[i+1,j]+cell_S_x_un[i-1,j]+cell_S_x_un[i,j+1]+cell_S_x_un[i,j-1]-4*cell_S_x_un[i,j])
#def Adv_y_n(i,j):
#    return (1/h)*((0.5*())**2-(0.5*())**2+(0.5*())*(0.5*())-(0.5*())*(0.5*()))
def Dif_y_n(i,j):
    return (1/(h**2))*(cell_S_y_vn[i+1,j]+cell_S_y_vn[i-1,j]+cell_S_y_vn[i,j+1]+cell_S_y_vn[i,j-1]-4*cell_S_y_vn[i,j])

def ref_vel_prof(x2,mu,dpdx,Lx2):
    '''
    function returning reference analytic sol
    '''
    return 1./(2.*mu)*dpdx*(x2**2 - (Lx2/2)**2)

def plt_sol(ifsaveplt):
    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(12, 12))
    fig.suptitle('iter= {:5d}, $L_2$= {:2.4e}'.format(n_iter, L_sq[0]))
    ax1.plot(cell_S_x_un[50, 1:Nx2 + 1], cell_S_x_coor_y[50, 1:Nx2 + 1], color='navy',
             label='numerical sol, $L^2$= {:10.4e}'.format(L_sq[0]))
    ax1.plot(ref_S_u[1:Nx2 + 1], cell_S_x_coor_y[50, 1:Nx2 + 1], color='red', label='analytical reference')
    ax1.set_xlabel('$u_1$ ($m/s$)')
    ax1.set_ylabel('$y$ (m)')
    ax1.legend()
    ax1.grid()

    #pcm = ax2.pcolormesh(cell_cent_x[1:Nx1 + 1, 1:Nx2 + 1], cell_cent_y[1:Nx1 + 1, 1:Nx2 + 1],
    #                     cell_S_x_un[1:Nx1 + 1, 1:Nx2 + 1], vmin=0.0, vmax=u1_max, cmap='inferno')
    pcm = ax2.pcolormesh(cell_cent_x[0:Nx1 + 2, 0:Nx2 + 2], cell_cent_y[0:Nx1 + 2, 0:Nx2 + 2],
                         cell_S_x_un[0:Nx1 + 2, 0:Nx2 + 2], vmin=0.0, vmax=u1_max, cmap='inferno')
    ax2.scatter(cell_cent_x[1:Nx1 + 1, 1:Nx2 + 1], cell_cent_y[1:Nx1 + 1, 1:Nx2 + 1], color='navy', s=20*(30/Nx2),
                label='cell center')
    #ax2.scatter(cell_cor_x[1:Nx1 + 2, 1:Nx2 + 2], cell_cor_y[1:Nx1 + 2, 1:Nx2 + 2], color='red', s=10*20*(30/Nx2),
    #            label='cell corner')
    cb=plt.colorbar(pcm, orientation='horizontal', aspect=70)
    cb.set_label('$u_1(m/s)$')
    ax2.set_xlabel('$x(m)$')
    ax2.set_ylabel('$y(m)$')

    plt.tight_layout()
    if(ifsaveplt==True):
        plt.savefig('./u_distr_'+str(Nx2)+'.png')
        plt.show()
    else:
        plt.show()

    return 0

def dump_sol():
    fname = 'sol_ny' + str(Nx2) + '.csv'
    y_coor = np.reshape(cell_S_x_coor_y[50, 1:Nx2 + 1], (-1, 1))
    u_ref = np.reshape(ref_S_u[1:Nx2 + 1], (-1, 1))
    u_sol = np.reshape(cell_S_x_un[50, 1:Nx2 + 1], (-1, 1))

    out_dat = np.concatenate((y_coor, u_ref, u_sol), axis=-1)
    np.savetxt('./' + fname, out_dat, delimiter=',')
    return 0

# =======================================================================================
#problem setup


# logistics
ifsaveplt = False
'''
node generation section
'''
#domain length

Lx2=0.02 #0.01
Lx1=2*Lx2 #0.04#0.02

#number of cells on each direction
Nx2=30
Nx1=2*Nx2 #60

#mesh spacing
h=Lx2/Nx2

#umax
u1_max = 0.01875 # m/s
u1_ave = 2./3.*u1_max
# properties
nu=1e-6
mu=1e-3
rho=1e+3


# numerical solver parameters
n_iter=0
infofreq= 1000
iofreq = 50000
max_iter = 20000
cfl_lim = 0.8
dt= cfl_lim*(Lx2/Nx2)/u1_max

# dpdx
gradP = lam_gradp(mu, Lx2, u1_max)



#cell centroid coor
#the +2 stands for ghost cells on each direction
cell_cent_x=np.zeros([Nx1+2,Nx2+2])
cell_cent_y=np.zeros([Nx1+2,Nx2+2])

cell_cent_un=np.zeros([Nx1+2,Nx2+2])
cell_cent_us=np.zeros([Nx1+2,Nx2+2])
cell_cent_unn=np.zeros([Nx1+2,Nx2+2])

#cell corner coor
cell_cor_x=np.zeros([Nx1+3,Nx2+3])
cell_cor_y=np.zeros([Nx1+3,Nx2+3])

#surf area of the cell
'''
S_x, S_y: cell "area" arrays
_coor_x, _coor_y: surface coordinate arrays
'''
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

#velocities at the pressure cell boundaries
'''
n: previous iter/initial
s: *, the predictor step
nn: n+1, with the div free correction
'''
cell_S_x_un=np.zeros([Nx1+2,Nx2+2])
cell_S_x_us=np.zeros([Nx1+2,Nx2+2])
cell_S_x_unn=np.zeros([Nx1+2,Nx2+2])

cell_S_x_vn=np.zeros([Nx1+2,Nx2+2])
cell_S_x_vs=np.zeros([Nx1+2,Nx2+2])
cell_S_x_vnn=np.zeros([Nx1+2,Nx2+2])

cell_S_y_un=np.zeros([Nx1+2,Nx2+2])
cell_S_y_us=np.zeros([Nx1+2,Nx2+2])
cell_S_y_unn=np.zeros([Nx1+2,Nx2+2])

cell_S_y_vn=np.zeros([Nx1+2,Nx2+2])
cell_S_y_vs=np.zeros([Nx1+2,Nx2+2])
cell_S_y_vnn=np.zeros([Nx1+2,Nx2+2])

#reference velocity profile
ref_S_u=np.zeros([Nx2+2])
L_sq=np.array([1.0,1.0]) # L2 residue, current and previous

#cell corner coor initialization
for j in range(0,Nx2+3):
    for i in range(0, Nx1+3):
        cell_cor_x[i,j]=(Lx1/Nx1)*(i-1)
        cell_cor_y[i,j]=-Lx2/2 + (Lx2/Nx2)*(j-1)
        
#cell cent coor storage
for j in range(0, Nx2+2):
    for i in range(0, Nx1+2):

        '''
        pressure cell:
        x----Sy----x(corners)
        |          |
        Sx  cent   Sx
        |          |
        x----Sy----x
        '''

        cell_cent_x[i,j]='{:10.6e}'.format(0.25*(cell_cor_x[i,j]+cell_cor_x[i+1,j]+cell_cor_x[i,j+1]+cell_cor_x[i+1,j+1]))
        cell_cent_y[i,j]='{:10.6e}'.format(0.25*(cell_cor_y[i,j]+cell_cor_y[i+1,j]+cell_cor_y[i,j+1]+cell_cor_y[i+1,j+1]))
        cell_S_x_coor_x[i,j]=(cell_cor_x[i,j]+cell_cor_x[i,j+1])/2
        cell_S_x_coor_y[i,j]=(cell_cor_y[i,j]+cell_cor_y[i,j+1])/2
        cell_S_y_coor_x[i,j]=(cell_cor_x[i,j]+cell_cor_x[i+1,j])/2
        cell_S_y_coor_y[i,j]=(cell_cor_y[i,j]+cell_cor_y[i+1,j])/2

        #initial conditions
        cell_S_x_un[i,j]=u1_ave #u1_max  #0.01
        cell_S_y_un[i,j]=0.00

        '''
        surface area arrays
        '''
        cell_S_x[i,j]=abs(cell_cor_y[i,j]-cell_cor_y[i,j+1])
        cell_S_y[i,j]=abs(cell_cor_x[i,j]-cell_cor_x[i+1,j])
        
        cell_S_x_nx[i,j]=(cell_cor_y[i,j+1]-cell_cor_y[i,j])/cell_S_x[i,j]
        cell_S_x_ny[i,j]=(cell_cor_x[i,j+1]-cell_cor_x[i,j])/cell_S_x[i,j]
        cell_S_y_nx[i,j]=(cell_cor_y[i+1,j]-cell_cor_y[i,j])/cell_S_y[i,j]
        cell_S_y_ny[i,j]=(cell_cor_x[i+1,j]-cell_cor_x[i,j])/cell_S_y[i,j]        



L_sq_r=L_sq[1]/L_sq[0]
for i in range(1, Nx2+1):
    ref_S_u[i]=ref_vel_prof(cell_S_x_coor_y[0,i], mu, gradP, Lx2)

# plot computational domain and initial solution
#plt_sol()

# the main loop
while (L_sq_r<=1.0 and n_iter<=max_iter):
    L_sq[0]=L_sq[1]

    # B.C. update
    # periodic
    #for j in range(0, Nx2+2):
    #    cell_S_x_un[0,j]=cell_S_x_un[-2,j]
    #    cell_S_x_un[-1,j]=cell_S_x_un[2,j]
    #cell_S_x_un[0, :] = cell_S_x_un[-2, :]
    cell_S_x_un[0, :] = cell_S_x_un[-3, :]
    cell_S_x_un[-1, :] = cell_S_x_un[2, :]
    # wall
    #for i in range(0, Nx1+2):
    #    cell_S_x_un[i,0]=-cell_S_x_un[i,1]
    #    cell_S_x_un[i,-1]=-cell_S_x_un[i,-2]
    cell_S_x_un[:, 0] = -cell_S_x_un[:, 1]
    cell_S_x_un[:, -1] = -cell_S_x_un[:, -2]

    #predictor step:
    for j in range(1, Nx2+1):
        for i in range(1, Nx1+1):
            cell_S_x_us[i,j]=cell_S_x_un[i,j]+dt*(-Adv_x_n(i,j)+nu*Dif_x_n(i,j))
            #cell_S_x_us[i,j]=cell_S_x_un[i,j]+dt*(nu*Dif_x_n(i,j))


    #B.C. update
    # periodic
    #cell_S_x_us[0, :] = cell_S_x_us[-2, :]
    cell_S_x_us[0, :] = cell_S_x_us[-3, :]
    cell_S_x_us[-1, :] = cell_S_x_us[2, :]
    # wall
    cell_S_x_us[:, 0] = -cell_S_x_us[:, 1]
    cell_S_x_us[:, -1] = -cell_S_x_us[:, -2]
            
    #corrector step:
    for j in range(1, Nx2+1):
        for i in range(1, Nx1+1):
            cell_S_x_unn[i,j]=cell_S_x_us[i,j]-(1/rho)*(dt)*(gradP)
                    
    #B.C. update        
    #for j in range(0, Nx2+2): # periodicity
    #    cell_S_x_unn[0,j]=cell_S_x_unn[-2,j]
    #    cell_S_x_unn[-1,j]=cell_S_x_unn[2,j]
    #cell_S_x_unn[0, :] = cell_S_x_unn[-2, :]
    cell_S_x_unn[0, :] = cell_S_x_unn[-3, :]
    cell_S_x_unn[-1, :] = cell_S_x_unn[2, :]

    #for i in range(0, Nx1+2): # wall
    #    cell_S_x_unn[i,0]=-cell_S_x_unn[i,1]
    #    cell_S_x_unn[i,-1]=-cell_S_x_unn[i,-2]
    cell_S_x_unn[:, 0] = -cell_S_x_unn[:, 1]
    cell_S_x_unn[:, -1] = -cell_S_x_unn[:, -2]

    #for j in range(1, Nx2+1): # reassign the solutions
    #    for i in range(1, Nx1+1):
    #        cell_S_x_un[i,j]=cell_S_x_unn[i,j]
    cell_S_x_un = cell_S_x_unn



    sq_sum_error=0 # initialize container for error
    
    for i in range(1,Nx2+1):
        sq_sum_error+=(ref_S_u[i]-cell_S_x_un[50,i])**2 # L2

    L_sq[1]=math.sqrt(sq_sum_error/(Nx2+1))

    if (n_iter%infofreq==0):
        print('iter #{:5d}, L2={:2.3e}'.format(n_iter, L_sq[1]))
    if (n_iter%iofreq==0):
        plt_sol(ifsaveplt=False)
    L_sq_r=L_sq[1]/L_sq[0]
    n_iter+=1


# final solution
plt_sol(ifsaveplt=True)

# dump solution to data file
dump_sol()

