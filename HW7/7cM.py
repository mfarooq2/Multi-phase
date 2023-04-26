# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 22:27:45 2023

@author:
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import array
import fractions    

def Adv_x(i,j):
    return (1/h) * ( ((u[i+1,j]+ u[i,j])/2)**2 - ((u[i,j] + u[i-1,j])/2)**2 )

def Dif_x(i,j):
    return (1/h**2) * (u[i+1,j]+u[i-1,j]+u[i,j+1]+u[i,j-1] - 4*u[i,j])

def anal(y):
    return -187.5*y**2 + 0.01875

def levelset (x,y):
    return radius - math.sqrt( (x-xd)**2 + (y - yd)**2 )  

def heavy (phi, M):
    if phi< -M*h:
        return 0
    elif abs(phi) <= M*h:
        return 0.5 * (1+phi/(M*h) + (1/math.pi)*math.sin(math.pi*phi/(M*h)))  
    elif phi > M*h:
        return 1
    
def L (u, phi_plus, phi_minus):  # Recall this is the ad hoc method
    return -u*(phi_plus - phi_minus)/(h)

        
# constants 
M = 3
nu = 1e-6
mul = 1e-3
mug = 1e-5
rhol = 1e3
rhog = 1
gradP = -0.375
max_vel = 0.01875 # this is the max velocity of the analytical solution

# building the domain
L1 = 0.04 # x
L2 = 0.02 # y
Ny = 50
Nx = int(Ny*2) 
h = L1/Nx

# Set up CFL and dt
cfl = 0.999
dt = cfl*h/max_vel

#droplet info
radius = L2/4
xd = 0.02
yd = 0.01

#Initialize the property grids
phi = np.zeros([Nx+2,Ny])
phi_star = np.zeros([Nx+2,Ny])
phi_n1 = np.zeros([Nx+2,Ny])
rho = np.zeros([Nx+2,Ny])
mu = np.zeros([Nx+2,Ny])
xlist = np.linspace(0,L1,Nx+2)
ylist = np.linspace(0,L2, Ny)   # we crop out two rows of nodes because those are the ghost nodes that give us no slip BC
grid_x = np.zeros([len(xlist),len(ylist)]) 
grid_y = np.zeros([len(xlist), len(ylist)])
D_plus_grid = np.zeros([len(xlist), len(ylist)])
D_minus_grid = np.zeros([len(xlist), len(ylist)])

#Now let's build an analytical velocity grid, then define a discretized analytical velocity profile
u = np.zeros([len(xlist), len(ylist)])
analist= np.linspace(-L2/2,L2/2,Ny)
for i in range(0, len(xlist)):
    for j in range(0, len(ylist)):
        u[i,j] = anal(analist[j])

#Now let's set up constant grids that will be out of the advection for loop
for i in range(0,len(xlist)):
    for j in range(0,len(ylist)):
        grid_y[i,:] = ylist
        grid_x[:,j] = xlist
        phi[i,j] = levelset(grid_x[i,j], grid_y[i,j])  #Now fill up phi, which in 7a was called level_set_field
        
# Okay big dog, from here on out everything will advected and updated

for iteration in range(0,100):
    for i in range(0,len(xlist)):
        for j in range(0,len(ylist)):
            rho[i,j] = rhol*heavy(phi[i,j], M) + rhog*(1-heavy(phi[i,j], M))
            mu[i,j]  = mul*heavy(phi[i,j], M) + mug*(1-heavy(phi[i,j], M))

            if i < len(xlist)-1:
                phi_star[i,j] = phi[i,j] + dt * L(u[i,j], phi[i+1,j],phi[i-1,j])
            else:
                phi_star[i,j] = phi[i,j] + dt * L(u[i,j], phi[0,j],phi[i-1,j])
     
    # Do the corrector step for level set field advection
    for i in range(0,len(xlist)):
        for j in range(0,len(ylist)):
            phi_n1[i,j] = phi[i,j] + dt/2 * (L(u[i,j],  phi[0,j], phi[i-1,j]) + L(u[i,j],  phi_star[0,j], phi_star[i-1,j]))
            
    figure, axes = plt.subplots()
    plt.contourf(grid_x, grid_y, phi, 10, cmap= 'gray')
    plt.title('Level Set Field when Ny = 50 \n timestep = %i' % iteration) 
    legend=plt.colorbar() 
    legend.ax.set_title('distance from interface',fontsize=8) 
    plt.xlabel('X') 
    plt.ylabel('Y') 
    #interface = plt.Circle((xd, yd), radius, color ='red', fill = False)
    #axes.add_artist(interface)
    plt.show()        

    # now update the level set field
    phi = phi_n1
