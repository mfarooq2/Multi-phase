from util_tools.operators import *
from numpy import linalg as LA
def predictor(un,vn, mu, rho,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    nx1=var_dict['nx1']
    nx2=var_dict['nx2']
    #nu=var_dict['nu']
    #gradP=var_dict['gradP']
    dt=var_dict['dt']
    gradP=-var_dict['gradP']
    us=np.zeros_like(un)
    us[1:,1:-1]=un[1:,1:-1]+dt*((mu/rho)*Dif_x_n(un,var_dict)-Adv_x_n(un,vn,var_dict))  #
    return us      
def projector( us, vs, pn,rho,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    nx1=var_dict['nx1']
    nx2=var_dict['nx2']
    dt=var_dict['dt']
    h=var_dict['h']
    pnn=np.zeros_like(pn)
    #rho=inter_polator(rho,var_dict)
        
    ##Edges
    #Left Edge
    pnn[ 0,1:-1]=((h)*((rho[ 0,1:-1]/(dt)) * (us[ 1,2:-2] - us[ 0,2:-2] )) - (pn[ 1,1:-1] + pn[0,1+1:] + pn[ 0,1-1:-1-1]))/(-3)   #pnn[ 0,1:-1]
    #Right Edge
    pnn[-1,1:-1]=((h)*((rho[-1,1:-1]/(dt)) * (us[-1,2:-2] - us[-2,2:-2] )) - (pn[-2,1:-1]+pn[-1,1+1:] + pn[-1,1-1:-1-1]))/(-3)

    #Bottom Edge
    pnn[1:-1,0]=((h)*((rho[1:-1,0]/(dt)) * (us[2:-1,1] - us[1:-2,1] )) - (pn[2:,0] + pn[0:-2,0] + pn[1:-1,1]))/(-3)  #  pnn[1:-1,0]
        #Corners

    ##Top-Left
    pnn[0,-1] = ((h) * ((rho[0,-1]/(dt))*(us[1,-2]-us[0,-2])) - (pn[1,-1]+ pn[0,-2])) / (-2)
    ##Bottom-Left
    pnn[0,0] = ((h) * ((rho[0,0]/(dt))*(us[1,1]-us[0,1])) - (pn[1,0]+ pn[0,1])) / (-2)



    ##Top-Right
    pnn[-1,-1] = ((h) * ((rho[-1,-1]/(dt))*(us[-1,-2]-us[-2,-2])) - (pn[-2,-1]+ pn[-1,-2])) / (-2)
    ##Bottom-Right
    pnn[-1,0] = ((h) * ((rho[-1,0]/(dt))*(us[-1,1]-us[-2,1])) - (pn[-2,0]+ pn[-1,1])) / (-2)
    
    
    
    
    #Top Edge
    pnn[1:-1,-1]=((h)*((rho[1:-1,-1]/(dt)) * (us[2:-1,-2] - us[1:-2,-2] )) - (pn[2:,-1]+pn[0:-2,-1]+pn[1:-1,-2]))/(-3)   ##pnn[1:-1,-1]





    ##Inner Domain
    pnn[1:-1,1:-1] = ((h) * ((rho[1:-1,1:-1]/(dt))*(us[2:-1,2:-2]-us[1:-2,2:-2])) - (pn[2:,1:-1]+ pn[0:-2,1:-1]+ pn[1:-1,2:]+ pn[1:-1,0:-2])) / (-4)


    return pnn
   
def corrector(us, pnn, unn, rho,var_dict):
    dt=var_dict['dt']
    h=var_dict['h']
    unn=np.zeros_like(us)
    unn[1:-1,1:-1] = us[1:-1,1:-1]-(1/(h*rho[1:,:])) * (dt) * (pnn[1:,:]- pnn[:-1,:])
    return unn


def phis_predictor(phin,un,vn,var_dict):
    Nx1=var_dict['Nx1']
    Nx1=var_dict['Nx2']
    nx1=var_dict['nx1']
    nx2=var_dict['nx2']
    dt=var_dict['dt']
    phis=np.zeros_like(phin)
    L_phi_n=L_phi(phin,un,vn,var_dict)
    phis=phin+dt*L_phi_n
    return L_phi_n,phis
        
def phinn_corrector(L_phi_n,phin,phis,un,vn,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    nx1=var_dict['nx1']
    nx2=var_dict['nx2']
    dt=var_dict['dt']
    phinn=np.zeros_like(phin)
    phinn=phin+0.5*dt*(L_phi_n+L_phi(phis,un,vn,var_dict))
    return phinn
def BC_looper(quant):
    
    quant[0,:]=quant[-2,:]
    quant[-1,:]=quant[1,:]
    
    quant[:,0]=-quant[:,1]
    quant[:,-1]=-quant[:,-2]
    return quant 