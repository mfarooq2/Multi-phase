import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt

L: float = 0.01
N_n: int = 60
r_dpl: float = 0.001
x_dpl: float = 0.005
h: float = L/N_n

rho_l: float = 1000
rho_g: float = 1

cell_x: np.ndarray = np.zeros(N_n)
cell_phi: np.ndarray = np.zeros(N_n)
cell_rho: np.ndarray = np.zeros(N_n)

def m_o1() -> float:
    return rho_l*h

def m_o2(rho: float) -> float:
    return rho*h

def m_o3(rho: float) -> float:
    w: float = (rho-rho_g)/(rho_l-rho_g)
    return rho_l*w*h

def hvsd(h: float, phi: float, M: int) -> float: # type: ignore
    if phi<-1*M*h:
        return 0
    elif abs(phi)<= M*h:
        tri=0.5*(1+phi/(M*h)+math.sin(math.pi*phi/(M*h))/math.pi)
        if tri is None:
            return 0
        else:
            return 0.5*(1+phi/(M*h)+math.sin(math.pi*phi/(M*h))/math.pi)
    elif phi> M*h:
        return 1

def lvlset(x: float) -> float:
    if x > (x_dpl-r_dpl) and x < (x_dpl+r_dpl):
        return min(abs(x-(x_dpl-r_dpl)), abs(x-(x_dpl+r_dpl)))
    else:
        return -1*min(abs(x-(x_dpl-r_dpl)), abs(x-(x_dpl+r_dpl)))

def calc_moments(M: int) -> tuple:
    m_1=0
    m_2=0
    m_3=0
    for i in range(len(cell_x)):
        cell_x[i]=0.0+i*h
        cell_phi[i]=lvlset(cell_x[i])
        cell_rho[i]=rho_l*hvsd(h,cell_phi[i], M)+rho_g*(1-hvsd(h,cell_phi[i], M))
        m_3+=m_o3(cell_rho[i])
        if cell_phi[i]>0:
            m_1+=m_o1()
            m_2+=m_o2(cell_rho[i])
    return (m_1, m_2, m_3)

def calc_moments_2(M: int) -> tuple:
    m_1=0
    m_2=0
    m_3=0
    for i in range(len(cell_x)):
        cell_x[i]=0.0+i*h
        m_1+=m_o1()
        m_2+=m_o2(rho_l*hvsd(h,lvlset(cell_x[i]), M)+rho_g*(1-hvsd(h,lvlset(cell_x[i]), M)))
        m_3+=m_o3(rho_l*hvsd(h,lvlset(cell_x[i]), M)+rho_g*(1-hvsd(h,lvlset(cell_x[i]), M)))
    return (m_1, m_2, m_3)

for M in range(1,4):
    print('m_1={:4.3e}, m_2={:4.3e}, m_3={:4.3e}, M={:1d}'.format(*calc_moments(M), M))
    plt.plot(cell_x, cell_rho, color=(0.7, 0.18*M+0.2, 0), label='M={:1d}'.format(M))
    plt.yscale('log')
    plt.legend()
    plt.title('domain density $(kg/m^3)$, droplet radius= {:4.3e} m'.format(r_dpl), fontsize=8)
    plt.grid()
    plt.savefig('hw5_3_density_distr.png')
    plt.show()