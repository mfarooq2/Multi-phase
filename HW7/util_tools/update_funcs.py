from util_tools.operators import *


def us_looper(cell_S_x_un, cell_cent_mu, cell_cent_rho, cell_S_x_us,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    nu=var_dict['nu']
    gradP=var_dict['gradP']
    rho=var_dict['rho']
    dt=var_dict['dt']
    
    cell_S_x_us[1:Nx1+1,1:Nx2+1]=cell_S_x_un[1:Nx1+1,1:Nx2+1]+dt*((nu)*Dif_x_n(cell_S_x_un,var_dict))+(-gradP)/rho
    return cell_S_x_us
def us_BC_looper(cell_S_x_us,var_dict):
    Nx2=var_dict['Nx2']
    Nx1=var_dict['Nx1']
    cell_S_x_us[0,:Nx2+2]=cell_S_x_us[-2,:Nx2+2]     
    cell_S_x_us[-1,:Nx2+2]=cell_S_x_us[1,:Nx2+2]
    cell_S_x_us[:Nx1+2,0]=cell_S_x_us[:Nx1+2,1]
    cell_S_x_us[:Nx1+2,-1]=cell_S_x_us[:Nx1+2,-2]
    return cell_S_x_us       
    
def p_looper( cell_S_x_us, cell_S_y_vs,cell_S_x_unn,cell_S_y_vnn, cell_cent_pn, cell_cent_pnn,var_dict):
    Nx1=var_dict['Nx1']
    Lx1=var_dict['Lx1']
    Nx2=var_dict['Nx2']
    Lx2=var_dict['Lx2']
    nu=var_dict['nu']
    rho=var_dict['rho']
    dt=var_dict['dt']
    cell_vol=var_dict['cell_vol']
    
    U_s=(rho/(dt*(Lx1/Nx1)))*(cell_S_x_us[2,2:Nx2]-cell_S_x_unn[1,2:Nx2]+cell_S_y_vs[1,2+1:Nx2+1]-cell_S_y_vs[1,2:Nx2])
    cell_cent_pnn[1,2:Nx2]=(cell_vol*U_s-(cell_cent_pn[2,2:Nx2]+cell_cent_pn[1,2+1:Nx2+1]+cell_cent_pn[1,2-1:Nx2-1]))/(-3)
    cell_cent_pn[1,2:Nx2]=cell_cent_pnn[1,2:Nx2]

    U_s=(rho/(dt*(Lx1/Nx1)))*(cell_S_x_us[2+1:Nx1+1,1:Nx2+1]-cell_S_x_us[2:Nx1,1:Nx2+1]+cell_S_y_vs[2:Nx1,1+1:Nx2+1+1]-cell_S_y_vs[2:Nx1,1:Nx2+1])
    cell_cent_pnn[2:Nx1, 1 : Nx2 + 1] = (
        cell_vol * U_s
        - (
            cell_cent_pn[2 + 1 : Nx1 + 1, 1 : Nx2 + 1]
            + cell_cent_pn[2 - 1 : Nx1 - 1, 1 : Nx2 + 1]
            + cell_cent_pn[2:Nx1, 1 + 1 : Nx2 + 1 + 1]
            + cell_cent_pn[2:Nx1, 0 : Nx2 + 1 - 1]
        )
    ) / -4
    cell_cent_pn[2:Nx1,1:Nx2+1]=cell_cent_pnn[2:Nx1,1:Nx2+1]

    U_s=(rho/(dt*(Lx1/Nx1)))*(cell_S_x_us[2,1]-cell_S_x_unn[1,1]+cell_S_y_vs[1,2]-cell_S_y_vnn[1,1])
    cell_cent_pnn[1,1]=(cell_vol*U_s-(cell_cent_pn[2,1]+cell_cent_pn[1,2]))/(-2)
    cell_cent_pn[1,1]=cell_cent_pnn[1,1]

    U_s=(rho/(dt*(Lx1/Nx1)))*(cell_S_x_unn[-2+1,1]-cell_S_x_us[-2,1]+cell_S_y_vs[-2,2]-cell_S_y_vnn[-2,1])
    cell_cent_pnn[-2,1]=(cell_vol*U_s-(cell_cent_pn[-2-1,1]+cell_cent_pn[-2,2]))/(-2)
    cell_cent_pn[-2,1]=cell_cent_pnn[-2,1]

    U_s=(rho/(dt*(Lx1/Nx1)))*(cell_S_x_us[2,-2]-cell_S_x_unn[1,-2]+cell_S_y_vnn[1,-2]-cell_S_y_vnn[1,-3])
    cell_cent_pnn[1,-2]=(cell_vol*U_s-(cell_cent_pn[2,-2]+cell_cent_pn[1,-3]))/(-2)
    cell_cent_pn[1,-2]=cell_cent_pnn[1,-2]

    U_s=(rho/(dt*(Lx1/Nx1)))*(cell_S_x_unn[-2+1,-2]-cell_S_x_us[-2,-2]+cell_S_y_vnn[-2,-2+1]-cell_S_y_vs[-2,-2])
    cell_cent_pnn[-2,-2]=(cell_vol*U_s-(cell_cent_pn[-2-1,-2]+cell_cent_pn[-2,-2-1]))/(-2)
    cell_cent_pn[-2,-2]=cell_cent_pnn[-2,-2]
    
    return cell_cent_pn

def p_BC_looper(cell_cent_pn,var_dict):
    Nx2=var_dict['Nx2']
    Nx1=var_dict['Nx1']
    dt=var_dict['dt']
    cell_cent_pn[0,:Nx2+2]=cell_cent_pn[-2,:Nx2+2]     
    cell_cent_pn[-1,:Nx2+2]=cell_cent_pn[1,:Nx2+2]         
    cell_cent_pn[:Nx1+2,0]=cell_cent_pn[:Nx1+2,-2]
    cell_cent_pn[:Nx1+2,-1]=cell_cent_pn[:Nx1+2,1]
    return cell_cent_pn
   
def unn_looper(cell_S_x_us, cell_cent_pnn, cell_S_x_unn, cell_cent_rho,var_dict):
    Nx2=var_dict['Nx2']
    Nx1=var_dict['Nx1']
    dt=var_dict['dt']
    cell_S_x_unn[1 : Nx2 + 1, 1 : Nx1 + 1] = cell_S_x_us[
        1 : Nx2 + 1, 1 : Nx1 + 1
    ] - (1 / cell_cent_rho[1 : Nx2 + 1, 1 : Nx1 + 1]) * (dt) * (
        cell_cent_pnn[1 : Nx2 + 1, 1 : Nx1 + 1]
        - cell_cent_pnn[0 : Nx2 + 1 - 1, 1 : Nx1 + 1]
    )
    return cell_S_x_unn
def unn_BC_looper(cell_S_x_unn,var_dict):
    Nx2=var_dict['Nx2']
    Nx1=var_dict['Nx1']
    cell_S_x_unn[0,:Nx2+2]=cell_S_x_unn[-2,:Nx2+2]     
    cell_S_x_unn[-1,:Nx2+2]=cell_S_x_unn[1,:Nx2+2]
    cell_S_x_unn[:Nx1+2,0]=cell_S_x_unn[:Nx1+2,1]
    cell_S_x_unn[:Nx1+2,-1]=cell_S_x_unn[:Nx1+2,-2]
    return cell_S_x_unn



def phis_looper(cell_cent_phis,cell_cent_phin,cell_S_x_un,cell_S_y_vn,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    dt=var_dict['dt']
    cell_cent_phis[1:Nx1+1,1:Nx2+1]=cell_cent_phin[1:Nx1+1,1:Nx2+1]+dt*L_phi_n(cell_cent_phis,cell_cent_phin,cell_S_x_un,cell_S_y_vn,var_dict)[1:Nx1+1,1:Nx2+1]
    return cell_cent_phis

def phis_BC_looper(cell_cent_phis,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    cell_cent_phis[0,:Nx2+2]=cell_cent_phis[-2,:Nx2+2]       
    cell_cent_phis[-1,:Nx2+2]=cell_cent_phis[1,:Nx2+2]
    cell_cent_phis[:Nx1+2,0]=cell_cent_phis[:Nx1+2,-2]
    cell_cent_phis[:Nx1+2,-1]=cell_cent_phis[:Nx1+2,1]
    return cell_cent_phis
        
def phinn_looper(cell_cent_phinn,cell_cent_phis,cell_cent_phi_ds,cell_cent_phin,cell_S_x_un,cell_S_x_us,cell_S_y_vn,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    dt=var_dict['dt']
    cell_cent_phinn[1:Nx1+1,1: Nx2+1]=cell_cent_phin[1:Nx1+1,1: Nx2+1]+0.5*dt*(L_phi_n(cell_cent_phis,cell_cent_phin,cell_S_x_un,cell_S_y_vn,var_dict)+L_phi_s(cell_cent_phis,cell_cent_phin,cell_S_x_us,cell_S_y_vn,var_dict))[1:Nx1+1,1: Nx2+1]
    return cell_cent_phinn
def phinn_BC_looper(cell_cent_phinn,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    cell_cent_phinn[0,:Nx2+2]=cell_cent_phinn[-2,:Nx2+2]
    cell_cent_phinn[-1,:Nx2+2]=cell_cent_phinn[1,:Nx2+2]
    
    cell_cent_phinn[:Nx1+2,0]=cell_cent_phinn[:Nx1+2,-2]
    cell_cent_phinn[:Nx1+2,-1]=cell_cent_phinn[:Nx1+2,1]  
    return cell_cent_phinn 
            
def phi_ds_looper(cell_cent_phin,cell_cent_phi_dn, cell_cent_phi_ds,var_dict):
    Nx1=var_dict['Nx1']                        
    Nx2=var_dict['Nx2']
    dtau=var_dict['dtau']
    cell_cent_phi_ds[1:Nx1+1,1: Nx2+1]=cell_cent_phi_dn[1:Nx1+1,1: Nx2+1]+dtau*L_phi_d_n(cell_cent_phin,cell_cent_phi_dn,var_dict)
    return cell_cent_phi_ds
def phi_ds_BC_looper(cell_cent_phi_ds,var_dict):        
    Nx1=var_dict['Nx1']                        
    Nx2=var_dict['Nx2']
    dtau=var_dict['dtau']
    cell_cent_phi_ds[0,:Nx2+2]=cell_cent_phi_ds[-2,:Nx2+2]
    cell_cent_phi_ds[-1,:Nx2+2]=cell_cent_phi_ds[1,:Nx2+2]
    cell_cent_phi_ds[:Nx1+2,0]=cell_cent_phi_ds[:Nx1+2,-2]
    cell_cent_phi_ds[:Nx1+2,-1]=cell_cent_phi_ds[:Nx1+2,1]    
    return cell_cent_phi_ds
    
    
def phi_dnn_looper(cell_cent_phi_dn, cell_cent_phin, cell_cent_phi_dnn,cell_cent_phi_ds,cell_cent_phis,var_dict): 
    Nx1=var_dict['Nx1']                        
    Nx2=var_dict['Nx2']
    dtau=var_dict['dtau']
    cell_cent_phi_dnn[1:Nx1+1,1:Nx2+1]=cell_cent_phi_dn[1:Nx1+1,1:Nx2+1]+0.5*dtau*(L_phi_d_n(cell_cent_phin,cell_cent_phi_dn,var_dict)+L_phi_ds(cell_cent_phi_ds,cell_cent_phis,var_dict))
    return cell_cent_phi_dnn
def phi_dnn_BC_looper(cell_cent_phi_dnn,var_dict):  
        Nx1=var_dict['Nx1']                        
        Nx2=var_dict['Nx2']
        cell_cent_phi_dnn[0,:Nx2+2]=cell_cent_phi_dnn[-2,:Nx2+2]
        cell_cent_phi_dnn[-1,:Nx2+2]=cell_cent_phi_dnn[1,:Nx2+2]
        cell_cent_phi_dnn[:Nx1+2,0]=cell_cent_phi_dnn[:Nx1+2,-2]
        cell_cent_phi_dnn[:Nx1+2,-1]=cell_cent_phi_dnn[:Nx1+2,1]
        return cell_cent_phi_dnn