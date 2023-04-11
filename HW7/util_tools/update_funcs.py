from util_tools.operators import *
from numpy import linalg as LA
def us_looper(un,vn, mu, rho, us,var_dict):
    
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    #nu=var_dict['nu']
    #gradP=var_dict['gradP']
    dt=var_dict['dt']
    gradP=-var_dict['gradP']
    us=un+dt*((mu/rho)*Dif_x_n(un,var_dict)-Adv_x_n(un,vn,var_dict))  #
    return us      
def p_looper( us, vs, pn,rho, pnn,var_dict):
    Nx1=var_dict['Nx1']
    Lx1=var_dict['Lx1']
    Nx2=var_dict['Nx2']
    dt=var_dict['dt']
    h=var_dict['h']
    pnn[1,2:Nx2]=((h**2)*((rho[1,2:Nx2]/(dt*(Lx1/Nx1)))*(us[2,2:Nx2]-us[1,2:Nx2]+us[1,2+1:Nx2+1]-vs[1,2:Nx2]))-(pn[2,2:Nx2]+pn[1,2+1:Nx2+1]+pn[1,2-1:Nx2-1]))/(-3)
    pnn[-2,2:Nx2]=((h**2)*((rho[-2,2:Nx2]/(dt*(Lx1/Nx1)))*(us[-1,2:Nx2]-us[-2,2:Nx2]+vs[-2,2+1:Nx2+1]-vs[-2,2:Nx2]))-(pn[-3,2:Nx2]+pn[-2,2+1:Nx2+1]+pn[-2,2-1:Nx2-1]))/(-3)
    pnn[2:Nx1,1:Nx2+1] = ((h**2) * ((rho[2:Nx1,1:Nx2+1]/(dt*(Lx1/Nx1)))*(us[2+1:Nx1+1,1:Nx2+1]-us[2:Nx1,1:Nx2+1]+vs[2:Nx1,1+1:Nx2+1+1]-vs[2:Nx1,1:Nx2+1])) - (pn[2+1:Nx1+1,1:Nx2+1]+ pn[2-1:Nx1-1,1:Nx2+1]+ pn[2:Nx1,1+1:Nx2+1+1]+ pn[2:Nx1, 0:Nx2+1-1])) / -4
    pnn[1,1]=((h**2)*((rho[1,1]/(dt*(Lx1/Nx1)))*(us[2,1]-us[1,1]+vs[1,2]-vs[1,1]))-(pn[2,1]+pn[1,2]))/(-2)
    pnn[-2,1]=((h**2)*((rho[-2,1]/(dt*(Lx1/Nx1)))*(us[-2+1,1]-us[-2,1]+vs[-2,2]-vs[-2,1]))-(pn[-2-1,1]+pn[-2,2]))/(-2)
    pnn[1,-2]=((h**2)*((rho[1,-2]/(dt*(Lx1/Nx1)))*(us[2,-2]-us[1,-2]+vs[1,-2]-vs[1,-3]))-(pn[2,-2]+pn[1,-3]))/(-2)
    pnn[-2,-2]=((h**2)*((rho[-2,-2]/(dt*(Lx1/Nx1)))*(us[-2+1,-2]-us[-2,-2]+vs[-2,-2+1]-vs[-2,-2]))-(pn[-2-1,-2]+pn[-2,-2-1]))/(-2)
        
        
    return pnn
   
def unn_looper(us, pnn, unn, rho,var_dict):
    Nx2=var_dict['Nx2']
    Nx1=var_dict['Nx1']
    dt=var_dict['dt']
    h=var_dict['h']
    unn[1:Nx1+1,1:Nx2+1] = us[1:Nx1+1,1:Nx2+1]-(1/(h*rho[1:Nx1+1,1:Nx2+1])) * (dt) * (pnn[1:Nx1+1,1:Nx2+1]- pnn[1-1:Nx1+1-1,1:Nx2+1])
    return unn

def phis_looper(phis,phin,un,vn,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    dt=var_dict['dt']
    phis=phin+dt*L_phi(phin,un,vn,var_dict)
    return phis
        
def phinn_looper(phinn,phis,phin,un,us,vs,vn,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    dt=var_dict['dt']
    phinn=phin+0.5*dt*(L_phi(phis,us,vs,var_dict)+L_phi(phin,un,vn,var_dict))
    return phinn
def BC_looper(quant,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    quant[0,0:Nx2+2]=quant[-2,0:Nx2+2]
    quant[-1,0:Nx2+2]=quant[1,0:Nx2+2]
    
    #phinn[0:Nx1+2,0]=-phinn[0:Nx1+2,1]
    quant[0:Nx1+2,1]=-quant[0:Nx1+2,0]
    #phinn[0:Nx1+2,-1]=-phinn[0:Nx1+2,-2]
    quant[0:Nx1+2,-2]=-quant[0:Nx1+2,-1]
    return quant 
            
def phi_ds_looper(phin,phi_dn, phi_ds,var_dict):
    Nx1=var_dict['Nx1']                        
    Nx2=var_dict['Nx2']
    dtau=var_dict['dtau']
    phi_ds[1:Nx1+1,1: Nx2+1]=phi_dn[1:Nx1+1,1: Nx2+1]+dtau*L_phi_d_n(phin,phi_dn,var_dict)
    return phi_ds       

    
    
def phi_dnn_looper(phi_dn, phin, phi_dnn,phi_ds,phis,var_dict): 
    Nx1=var_dict['Nx1']                        
    Nx2=var_dict['Nx2']
    dtau=var_dict['dtau']
    phi_dnn[1:Nx1+1,1:Nx2+1]=phi_dn[1:Nx1+1,1:Nx2+1]+0.5*dtau*(L_phi_d_n(phin,phi_dn,var_dict)+L_phi_ds(phi_ds,phis,var_dict))
    return phi_dnn