import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

Lx = 0.04
Ly = 0.02

rd = (1/2)*Ly/2

ny = 30
nx = 2*ny
dx = Lx/nx
dy = Ly/ny
h = dx
xd = 0.02
L_advect = 0.008

x = np.linspace(0+ dx/2, Lx- dx/2, nx) # Grid points in x-direction
y = np.linspace(-0.01+ dy/2, 0.01- dy/2, ny) # Grid points in y-direction
X, Y = np.meshgrid(x, y)

# Define time parameters
cfl_lim = 0.05
umax = 0.01875
dt= cfl_lim*(h/umax)
N_advect = int(L_advect/(umax*dt))
l2 = 1.0 # L2 norm initialization
n = 0 # iteration 
ub = 0.0  
epsilon = 1e-3
rhol = 1000
mul = 1e-3
rhog = 1
mug = 1e-5

phi = np.zeros((ny, nx+4))
sgnphi = np.zeros((ny,nx+4))
f  = np.zeros((ny, nx))
rho = np.zeros((ny, nx))
mu = np.zeros((ny, nx))
dpdx = np.zeros((ny, nx))
M = 3

# Define velocity and pressure fields
u = np.zeros((ny+2, nx+2))  # u-velocity at cell edges in x-direction
v = np.zeros((ny+1, nx))  # v-velocity at cell edges in y-direction
p = np.zeros((ny, nx+2))  # pressure at cell centers
p_update = np.zeros((ny, nx+2)) 
ue =np.zeros((ny, nx))  # Exact velocity
A = u.copy() 
D = u.copy() 
u_star = u.copy()   # Intermediate u velocity in x-direction
v_star = v.copy()   # Intermediate v velocity in y-direction

# analytical solution

for j in range(0, ny):
    ue[j, :] = -187.5*y[j]**2 + 0.01875   

# initial condition
u[1:ny+1, 1:nx+1] = ue
u_star[1:ny+1, 1:nx+1] = ue 

            
# pressure initialization
p[:, :] = - 0.375*dx     

uc = np.zeros((ny, nx+2))
phi_star = np.zeros((ny, nx+4))
phi_update = np.zeros((ny, nx+4))
phi_phalf= np.zeros((ny, nx+4))
phi_mhalf= np.zeros((ny, nx+4))
Lphi= np.zeros((ny, nx+4))
Dxp = np.zeros((ny, nx))
Dxm = np.zeros((ny, nx))
Dxpp1 = np.zeros((ny, nx))
Dxmp1 = np.zeros((ny, nx))
Dxpm1 = np.zeros((ny, nx))
Dxmm1 = np.zeros((ny, nx))

 
# Level set distance function
for j in range(0, ny):
    for i in range(2, nx+2):
        phi[j, i] = rd -  np.sqrt((x[i-2]-xd)**2+(y[j])**2)

phi[:, 0] = phi[:, nx]  # Left wall
phi[:, 1] = phi[:, nx+1]  # Left wall
phi[:, nx+3] = phi[:, 2]  # Right wall
phi[:, nx+2] = phi[:, 3]  # Right wall

# # Create the plot
# plt.style.use("default")
# plt.contourf(X, Y, phi[:, 2:nx+2], 500, cmap='magma')
# plt.gca().set_aspect(1/1)
# plt.xlabel('x')
# plt.ylabel('y')
# plt.title('2D contour of level set function')
# plt.colorbar()
# plt.savefig('line_contour_level_Set.png', bbox_inches = 'tight', dpi=100)
# plt.show()
def property_update(phi_loc):
    threshold = M * h
    positive_mask = (phi_loc >= threshold)
    negative_mask = (phi_loc <= -threshold)
    sgnphi = np.sign(phi_loc)
    sgnphi[positive_mask] = 1
    sgnphi[negative_mask] = -1
    sgnphi[~(positive_mask | negative_mask)] = (
        phi_loc[~(positive_mask | negative_mask)] / (M * h) 
        - (1/np.pi) * np.sin(np.pi * phi_loc[~(positive_mask | negative_mask)] / (M * h))
    )
    # Define the threshold value for phi 
    threshold = M * h
    # Compute f for each point using numpy element-wise operations
    f = np.zeros((ny, nx))
    phi_loc_shifted = phi_loc[:, 2:-2] # shift phi_loc by 2 columns to match f shape

    negative_mask = (phi_loc_shifted < -threshold)
    within_threshold_mask = (np.abs(phi_loc_shifted) <= threshold)

    f[within_threshold_mask] = 0.5 * (1 + phi_loc_shifted[within_threshold_mask] / (M * h)
                                     + (1/np.pi) * np.sin(np.pi * phi_loc_shifted[within_threshold_mask] / (M * h)))
    f[negative_mask] = 0
    f[~(within_threshold_mask | negative_mask)] = 1
    
    
    # Compute rho for each point using numpy element-wise operations
    rho = rhol * f + rhog * (1 - f)
    # Compute mu for each point using numpy element-wise operations
    mu = mul * f + mug * (1 - f)
    nu = mu / rho   
    
    dpdx = -umax * mu * (2/(0.01)**2)
    
    return nu, dpdx, rho, mu, sgnphi

# Define functions for velocity and pressure updates
def inermediate_velocity_update(u, v, dt, dx, dy, rho, nu,p):
    
    # Create a copy of u
    un = u.copy()

    # Calculate A
    A[:, 1:-1] = (1/dx) * (((un[:, 2:] + un[:, 1:-1])/2)**2 - ((un[:, 1:-1] + un[:, :-2])/2)**2)
    
    # Calculate D
    D[1:-1, 1:-1] = (1/(dx*dy)) * (un[2:, 1:-1] - 4*un[1:-1, 1:-1] + un[:-2, 1:-1] + un[1:-1, 2:] + un[1:-1, :-2])
    # Update intermediate velocities
    u_star[1:ny+1, 1:nx+1] = un[1:ny+1, 1:nx+1] - dt*A[1:ny+1, 1:nx+1] + nu[0:ny, 0:nx] * dt * D[1:ny+1, 1:nx+1]
    return u_star

def pressure_update(u_star, v_star, dt, dx, dy, rho, epsilon,p):

    p_temp = p
    tolerance = 1
    
    while (tolerance > epsilon):
       p_update[0, 1:nx+1] = (1/3) * (p_temp[1, 1:nx+1] + p_temp[0, 2:nx+2] + p_temp[0, :nx]) \
                        - (dx/3) * (rho[0, :nx] / dt) * (u_star[1, 2:nx+2] - u_star[1, 1:nx+1])
       p_update[ny-1, 1:nx+1] = (1/3) * (p_temp[ny-2, 1:nx+1] + p_temp[ny-1, 2:nx+2] + p_temp[ny-1, :nx]) \
                                                    - (dx/3) * (rho[ny-1, :nx] / dt) * (u_star[ny, 2:nx+2] - u_star[ny, 1:nx+1])
       j, i = np.meshgrid(np.arange(1, ny-1), np.arange(1, nx+1), indexing='ij')
       p_update[j, i] = ((1/4)*(p_temp[j+1,i]+p_temp[j-1,i]+p_temp[j,i+1]+p_temp[j,i-1])) - \
                 ((dx/4)*(rho[j, i-1]/dt)*(u_star[j+1,i+1] - u_star[j+1,i]))
       tolerance = (abs(p_temp[:,1:nx+1] - p_update[:,1:nx+1])).max()
       print('Tolerance: ',tolerance)
       
       p_update[:, nx+1] = p_update[:,2]
       p_update[:, 0] = p_update[:,nx-1]
       p_temp = p_update.copy()
    return p_update

def velocity_update(u_star, v_star, dt, dx, dy, rho, p_new):
    
    # Update velocities
    # Update u-velocity at cell edges in x-direction
    u[1:ny+1, 1:nx+1] = u_star[1:ny+1, 1:nx+1] - (1/rho[0:ny, 0:nx])*(dt)* (p_new[0:ny, 0:nx]- p_new[0:ny, 0:nx])
    #dpdx[0:ny, 0:nx]
    return u

def switch(a, b):
    return a if abs(a) < abs(b) else b
    
def Lphi_calculation(phi_loc, uc_loc):
    
    Dxp = phi_loc[:, 2:] - phi_loc[:, 1:-1]
    Dxm = phi_loc[:, 1:-1] - phi_loc[:, :-2]

    Dxpp1 = phi_loc[:, 3:] - phi_loc[:, 2:-1]
    Dxmp1 = phi_loc[:, 2:-1] - phi_loc[:, 1:-2]


    for j in range(0, ny):
        for i in range(2, nx+2):
            if 0.5*(uc_loc[j, i-1] + uc_loc[j, i-2]) > 0:
                  phi_phalf[j,i]= phi_loc[j,i] + 0.5*switch(Dxp[j,i-2], Dxm[j,i-2])
            else:
                phi_phalf[j,i]= phi_loc[j,i+1] - 0.5*switch(Dxpp1[j,i-2], Dxmp1[j,i-2])

    Dxpm1 = phi_loc[:, 2:] - phi_loc[:, 1:-1]
    Dxmm1 = phi_loc[:, 1:-1] - phi_loc[:, :-2]
    for j in range(0, ny):
        for i in range(2, nx+2):
            if 0.5*(uc_loc[j, i-3] + uc_loc[j, i-2]) > 0:
                  phi_mhalf[j,i]= phi_loc[j,i-1] + 0.5*switch(Dxpm1[j,i-2], Dxmm1[j,i-2])
            else:
                  phi_mhalf[j,i]= phi_loc[j,i] - 0.5*switch(Dxp[j,i-2], Dxm[j,i-2])
    BC_looper(phi_phalf)
    BC_looper(phi_mhalf)
    Lphi[:, 2:nx+2] = -uc_loc[:, :nx]*((phi_phalf[:, 2:nx+2] - phi_mhalf[:, 2:nx+2])/h)
    BC_looper(Lphi)
    return Lphi


# TODO Rename this here and in `Lphi_calculation`
def BC_looper(quant):
    quant[:, 0] = quant[:, nx]
    quant[:, 1] = quant[:, nx+1]
    quant[:, nx+3] = quant[:, 2]
    quant[:, nx+2] = quant[:, 3]

# Run simulation in iteration
while (n < N_advect):
    nu, dpdx, rho, mu, sgnphi = property_update(phi)
    # Solve Navier-Stokes equations
    u_star = inermediate_velocity_update(u, v, dt, dx, dy, rho, nu,p)

    u_star[:, 0] = u_star[:, nx-1]  # Left wall
    u_star[:, nx+1] = u_star[:, 2]  # Right wall
    u_star[0, :] = 2*ub - u_star[1, :]  # Bottom wall
    u_star[ny+1, :] = 2*ub - u_star[ny, :]  # Top wall

    p_new = pressure_update(u_star, v_star, dt, dx, dy, rho, epsilon,p)

    p_new[:, nx+1] = p_new[:,2]
    p_new[:, 0] = p_new[:,nx-1]
    # print(p)

    u_updated = velocity_update(u_star, v_star, dt, dx, dy, rho, p_new)

    # B.C. for updated velocity
    u_updated[:, 0] = u_updated[:, nx-1]  # Left wall
    u_updated[:, nx+1] = u_updated[:, 2]  # Right wall
    u_updated[0, :] = 2*ub - u_updated[1, :]  # Bottom wall
    u_updated[ny+1, :] = 2*ub - u_updated[ny, :]  # Top wall

    u = u_updated

    # L2 norm calculation
    l2 = (norm(abs(u[1:ny,0]-ue[1:ny,0])))/ny

    # B.C. for updated velocity
    u[:, 0] = u[:, nx-1]  # Left wall
    u[:, nx+1] = u[:, 2]  # Right wall
    u[0, :] = 2*ub - u[1, :]  # Bottom wall
    u[ny+1, :] = 2*ub - u[ny, :]  # Top wall

    uc[:,1:-1] = 0.5 * (u[1:-1,2:] + u[1:-1,1:-1])

    uc[:, 0] = uc[:, nx-1]  # Left wall
    uc[:, nx+1] = uc[:, 2]  # Right wall

    Lphi = Lphi_calculation(phi, uc)
    j, i = np.meshgrid(np.arange(ny), np.arange(2, nx+2), indexing='ij')
    phi_star[j, i] = phi[j, i] + dt*Lphi[j,i]


    phi_star[:, 0] = phi_star[:, nx]  # Left wall
    phi_star[:, 1] = phi_star[:, nx+1]  # Left wall
    phi_star[:, nx+3] = phi_star[:, 2]  # Right wall
    phi_star[:, nx+2] = phi_star[:, 3]  # Right wall

    Lphi_update = Lphi_calculation(phi_star, uc)
    phi_update[:, 2:nx+2] = phi[:, 2:nx+2] + (dt/2) * (Lphi[:, 2:nx+2] + Lphi_update[:, 2:nx+2])
    phi_update[:, 0] = phi_update[:, nx]  # Left wall
    phi_update[:, 1] = phi_update[:, nx+1]  # Left wall
    phi_update[:, nx+3] = phi_update[:, 2]  # Right wall
    phi_update[:, nx+2] = phi_update[:, 3]  # Right wall
    phi = phi_update

    n = n+1