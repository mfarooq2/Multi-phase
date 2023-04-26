# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 18:16:26 2023

@author: 
"""

import math
import numpy as np
import matplotlib.pyplot as plt
import numpy.random as rdm

level = 2.0

xd = np.zeros(5)
yd = np.zeros(5)
xb = np.zeros(5)
yb = np.zeros(5)
rb = 0.25
rd = 0.25
level_set_field = np.zeros([151,76])

for i in range (0,5):
    xd[i] = rdm.uniform(0, 10)
    yd[i] = rdm.uniform(2, 5)
    xb[i] = rdm.uniform(0, 10)
    yb[i] = rdm.uniform(0, 2)
    
def levelset (x,y):
    def droplet(x,y,i):
        return rd - math.sqrt( (x-xd[i])**2 + (y - yd[i])**2 )   
    
    def bubble(x,y,i):
        return math.sqrt( (x - xb[i])**2 + (y - yb[i])**2 ) - rb

    if y > level: # this is the vapor region
        return max(level-y, droplet(x,y,0), droplet(x,y,1), droplet(x,y,2), droplet(x,y,3), droplet(x,y,4))

    else: # this is the liquid region
        return min(level-y, bubble(x,y,0), bubble(x,y,1),bubble(x,y,2),bubble(x,y,3),bubble(x,y,4))

grid_x = np.zeros([151,76]) 
grid_y = np.zeros([151, 76])

for i in range(0,151):
    for j in range(0,76):
        grid_x[i,j] = 0 + i*10/150
        grid_y[i,j] = 0 + j*5/75
        level_set_field[i,j] = levelset(grid_x[i,j], grid_y[i,j])        


plt.contourf(grid_x,grid_y,level_set_field,20, vmin=-4, vmax=4,cmap= 'viridis')
plt.title('Level Set Field') 
 

legend=plt.colorbar() 
legend.ax.set_title('distance from interface',fontsize=8) 
 

plt.xlabel('X') 
plt.ylabel('Y') 
plt.show()

# plt.plot(grid_y[100, :],level_set_field[100,:], color='red' )
# plt.show()



