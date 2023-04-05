# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 12:14:48 2020

@author: Wazowskai
"""

import numpy as np
import math
import matplotlib.pyplot as plt

'''
computational setup
'''
n_p=1000

'''
problem constants
'''
#density in kg/m^3
rho_l=998.21
rho_g=1.205
mu_l=1e-3
mu_g=1e-5
gamma=7e-3
g=9.8
#cd=0.47
#bubble diameter in m


t=np.zeros(n_p)
dt=0.0006/n_p
v_b=np.zeros(n_p)
a_b=np.zeros(n_p)


#def b_acc(i):
#    return ((rho_g-rho_l)*Vol_b*g)/m_b
    #return ((rho_g-rho_l)*Vol_b*g+(-0.125*rho_l*cd*abs(v_b[i-1])*v_b[i-1]*a_x_b))/m_b
#def F_d(i):
#    return 0.125*cd*rho_l*(v_b[i]**2)*a_x_b

def cd(i):
    re=rho_l*v_b[i]*(2*r_b)/mu_l
    eo=(rho_l-rho_g)*g*((2*r_b)**2)/gamma
    if re==0:
        return 0
    else:
        return math.sqrt(((16/re)*(1+(2)/(1+(16/re)+(3.315/re**0.5))))**2+((4*eo)/(eo+9.5))**2)
    #return math.sqrt(((16/re)*(1+(2)/(1+(16/re)+(3.315/re**0.5))))**2+((4*eo)/(eo+9.5))**2)
r_b_list=[5e-4, 6e-4, 7e-4, 8e-4, 9e-4, 1e-3, 1.5e-3,2.5e-3]
for r_b in r_b_list: 
    Vol_b=(4/3)*math.pi*(r_b)**3
    a_x_b=math.pi*(r_b)**2
    m_b=rho_g*Vol_b
    for i in range(1, len(t)):
        t[i]=dt*i
        v_b[i]=v_b[i-1]+dt*a_b[i-1]
        F_g=rho_g*Vol_b*g
        F_b=(rho_l)*Vol_b*g
        F_d=0.125*cd(i)*rho_l*abs(v_b[i])*v_b[i]*a_x_b
        #a_b[i]=(F_g-F_d)/m_b
        a_b[i]=(F_b-F_g-F_d)/m_b
    plt.plot(t, v_b, color=(r_b/2.5e-3,0.5,0), label='$r_{bub}$ '+str('{:4.1e}'.format(r_b))+' m')
    plt.legend(fontsize=8)
plt.grid()
plt.title('bubble $v_t$ trend, variable $C_D=fn(Re_b, Eo)$')
plt.ylabel('bubble velocity $v_b$ (m/s)')
plt.xlabel('time (s)')
plt.show()
