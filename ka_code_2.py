Code for #5.3
import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
L=0.01 N_n=60 r_dpl=0.001 x_dpl=0.005 global h h=L/N_n
rho_l=1000 rho_g=1
m_1=0 m_2=0 m_3=0
cell_x=np.zeros(N_n) cell_phi=np.zeros(N_n) cell_rho=np.zeros(N_n) def m_o1():
    return rho_l*h
def m_o2(rho):     return rho*h
def m_o3(rho): w=(rho-rho_g)/(rho_l-rho_g)     return rho_l*w*h
def lvlset(x):
if x > (x_dpl-r_dpl) and x < (x_dpl+r_dpl):
    return min(abs(x-(x_dpl-r_dpl)), abs(x-(x_dpl+r_dpl))) else:
    return-1*min(abs(x-(x_dpl-r_dpl)), abs(x-(x_dpl+r_dpl))) def hvsd(phi, M):
if phi<-1*M*h:

    return 0
elif abs(phi)<= M*h:
    return 0.5*(1+phi/(M*h)+math.sin(math.pi*phi/(M*h))/math.pi) elif phi> M*h:
    return 1 for M in range(1,4):
m_1=0
m_2=0
m_3=0
for i in range(0, len(cell_x)):
cell_x[i]=0.0+i*h
cell_phi[i]=lvlset(cell_x[i])
cell_rho[i]=rho_l*hvsd(cell_phi[i], M)+rho_g*(1-hvsd(cell_phi[i], M)) m_3+=m_o3(cell_rho[i])
if cell_phi[i]>0:
m_1+=m_o1()
m_2+=m_o2(cell_rho[i])
print('{:4.3e}, {:4.3e}, {:4.3e}, {:1d}'.format(m_1, m_2, m_3, M)) #plt.plot(cell_x, cell_phi, color='red', label='domain level-set')
plt.plot(cell_x, cell_rho, color=(0.7, 0.18*M+0.2, 0), label='M={:1d}'.format(M))
plt.yscale('log')
plt.legend()
plt.title('domain density $(kg/m^3)$, droplet radius= {:4.3e} m'.format(r_dpl), fontsize=8) plt.grid()
plt.savefig('hw5_3_density_distr.png')
plt.show()
#plt.plot(cell_x, cell_rho, color='navy', label='domain density $(kg/m^3)$')
#plt.show()