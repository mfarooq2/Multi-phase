import math
import numpy as np



def Adv_x_n(cell_S_x_un,i,j):
    return (1/h)*((0.5*(cell_S_x_un[i,j]+cell_S_x_un[i+1,j]))**2-(0.5*(cell_S_x_un[i,j]+cell_S_x_un[i-1,j]))**2+(0.5*(cell_S_x_un[i,j+1]+cell_S_x_un[i,j]))*(cell_S_y_vn[i,j+1]+cell_S_y_vn[i+1,j+1])-(0.5*(cell_S_x_un[i,j]+cell_S_x_un[i,j-1]))*(0.5*(cell_S_y_vn[i,j]+cell_S_y_vn[i+1,j]))) 

def Dif_x_n(cell_S_x_un,i,j):
    return (1/(h**2))*(cell_S_x_un[i+1,j]+cell_S_x_un[i-1,j]+cell_S_x_un[i,j+1]+cell_S_x_un[i,j-1]-4*cell_S_x_un[i,j])

def Dif_y_n(cell_S_y_vn,i,j):
    return (1/(h**2))*(cell_S_y_vn[i+1,j]+cell_S_y_vn[i-1,j]+cell_S_y_vn[i,j+1]+cell_S_y_vn[i,j-1]-4*cell_S_y_vn[i,j])

def ref_vel_prof(x2):
    '''
    function returning reference analytic sol
    '''
    return -1200*((x2-0.005)**2)+0.03

def lvlset_init(x,y,r_dpl):
    def ls_dpl(x,y):
        return -1*(math.sqrt((x-0.01)**2+(y-0.005)**2)-r_dpl)
    return ls_dpl(x,y)

def M_sw(a,b):
    return a if abs(a)<abs(b) else b
#level-set functions
                                                                                #Differences for L_phi_n #Operator Defined in 3.43
def D_x_p_n(cell_cent_phin,i,j):                                                             #Eq 3.46 Implementing Differences
    if i==Nx1+1:
        return cell_cent_phin[2,j]-cell_cent_phin[i,j]                      
    else:    
        return cell_cent_phin[i+1,j]-cell_cent_phin[i,j]                        #Dx_plus

def D_x_m_n(cell_cent_phin,i,j):
    return cell_cent_phin[i,j]-cell_cent_phin[i-1,j]                            #Dx_minus

def D_y_p_n(cell_cent_phin,i,j):
    return cell_cent_phin[i,j+1]-cell_cent_phin[i,j]                            #Dy_plus

def D_y_m_n(cell_cent_phin,i,j):
    return cell_cent_phin[i,j]-cell_cent_phin[i-1,j]                            #Dy_minus

                                                                                #Differences for L_phi_star #Operator Defined in 3.43                        
def D_x_p_s(cell_cent_phis,i,j):                                                               #Eq 3.46 Implementing Differences
    if i==Nx1+1:
        return cell_cent_phis[2,j]-cell_cent_phis[i,j]                          #Dx_plus
    else:                                                                       
        return cell_cent_phis[i+1,j]-cell_cent_phis[i,j]                     

def D_x_m_s(cell_cent_phis,i,j):       
    return cell_cent_phis[i,j]-cell_cent_phis[i-1,j]                            #Dx_minus

def D_y_p_s(cell_cent_phis,i,j):                                                               #Dy_plus
    return cell_cent_phis[i,j+1]-cell_cent_phis[i,j]                   

def D_y_m_s(cell_cent_phis,i,j):                                                               #Dy_minus
    return cell_cent_phis[i,j]-cell_cent_phis[i-1,j]

def L_phi_n(cell_S_x_unn,cell_cent_phin,i,j):                                                               #Eq 3.43 L_phi Operator

    def phi_xh_n(i,j):                                                          #For i+1/2,j Eq 3.44
        if 0.5*(cell_S_x_unn[i+1,j]+cell_S_x_unn[i,j])>0:
            return cell_cent_phin[i,j]+0.5*M_sw(D_x_p_n(i,j), D_x_m_n(i,j))
        elif 0.5*(cell_S_x_unn[i+1,j]+cell_S_x_unn[i,j])<0:
            return cell_cent_phin[i+1,j]-0.5*M_sw(D_x_p_n(i+1,j), D_x_m_n(i+1,j))
            
    def phi_hx_n(i,j):                                                          #For i-1/2,j Eq 3.44
        if 0.5*(cell_S_x_unn[i,j]+cell_S_x_unn[i-1,j])<0:
            return cell_cent_phin[i,j]-0.5*M_sw(D_x_p_n(i,j), D_x_m_n(i,j)) 
        elif 0.5*(cell_S_x_unn[i,j]+cell_S_x_unn[i-1,j])>0:
            return cell_cent_phin[i-1,j]+0.5*M_sw(D_x_p_n(i-1,j), D_x_m_n(i-1,j))
        
    # def phi_yh_n(i,j):                                                          #For i,j+1/2 Eq 3.44
    #     if 0.5*(cell_S_y_vnn[i,j+1]+cell_S_y_vnn[i,j])>0:
    #         return cell_cent_phin[i,j]+0.5*M_sw(D_y_p_n(i.j), D_y_m_n(i,j))
    #     elif 0.5*(cell_S_y_vnn[i,j+1]+cell_S_y_vnn[i,j])<0:
    #         return cell_cent_phin[i,j+1]+0.5*M_sw(D_y_p_n(i,j+1), D_y_m_n(i,j+1))
        
    # def phi_hy_n(i,j):                                                          #For i,j-1/2 Eq 3.44
    #     if 0.5*(cell_S_y_vnn[i,j]+cell_S_y_vnn[i,j-1])<0:
    #         return cell_cent_phin[i,j]-0.5*M_sw(D_y_p_n(i,j), D_y_m_n(i,j))
    #     elif 0.5*(cell_S_y_vnn[i,j]+cell_S_y_vnn[i,j-1])>0:
    #         return cell_cent_phin[i,j-1]+0.5*M_sw(D_y_p_n(i,j-1), D_y_m_n(i,j-1))
   
    #return -1*(cell_S_x_unn[i,j])*(phi_xh_n(i,j)-phi_hx_n(i,j))/h-(cell_S_y_vnn[i,j])*(phi_yh_n(i,j)-phi_hy_n(i,j))/h
    return -1*(cell_S_x_unn[i,j])*(phi_xh_n(i,j)-phi_hx_n(i,j))/h               #Eq 3.43 (L_phi_n)


def L_phi_s(cell_S_x_unn,cell_cent_phis,i,j):                                                               #Eq 3.43 L_phi_star Operator

    def phi_xh_s(i,j):                                                          #For i+1/2,j Eq 3.44
        if 0.5*(cell_S_x_unn[i+1,j]+cell_S_x_unn[i,j])>0:
            return cell_cent_phis[i,j]+0.5*M_sw(D_x_p_s(i,j), D_x_m_s(i,j))     
        elif 0.5*(cell_S_x_unn[i+1,j]+cell_S_x_unn[i,j])<0:
            return cell_cent_phis[i+1,j]-0.5*M_sw(D_x_p_s(i+1,j), D_x_p_s(i+1,j))
            
    def phi_hx_s(i,j):                                                          #For i-1/2,j Eq 3.44
        if 0.5*(cell_S_x_unn[i,j]+cell_S_x_unn[i-1,j])<0:
            return cell_cent_phis[i,j]-0.5*M_sw(D_x_p_s(i,j), D_x_m_s(i,j)) 
        elif 0.5*(cell_S_x_unn[i,j]+cell_S_x_unn[i-1,j])>0:
            return cell_cent_phis[i-1,j]+0.5*M_sw(D_x_p_s(i-1,j), D_x_m_s(i-1,j))
        
    # def phi_yh_s(i,j):                                                          #For i,,j+1/2 Eq 3.44
    #     if 0.5*(cell_S_y_vnn[i,j+1]+cell_S_y_vnn[i,j])>0:
    #         return cell_cent_phis[i,j]+0.5*M_sw(D_y_p_s(i.j), D_y_m_s(i,j))
    #     elif 0.5*(cell_S_y_vnn[i,j+1]+cell_S_y_vnn[i,j])<0:
    #         return cell_cent_phis[i,j+1]+0.5*M_sw(D_y_p_s(i,j+1), D_y_m_s(i,j+1))
        
    # def phi_hy_s(i,j):                                                          #For i,j-1/2 Eq 3.44
    #     if 0.5*(cell_S_y_vnn[i,j]+cell_S_y_vnn[i,j-1])<0:
    #         return cell_cent_phis[i,j]-0.5*M_sw(D_y_p_s(i,j), D_y_m_s(i,j))
    #     elif 0.5*(cell_S_y_vnn[i,j]+cell_S_y_vnn[i,j-1])>0:
    #         return cell_cent_phis[i,j-1]+0.5*M_sw(D_y_p_s(i,j-1), D_y_m_s(i,j-1))
    #return -1*(cell_S_x_unn[i,j])*(phi_xh_s(i,j)-phi_hx_s(i,j))/h-(cell_S_y_vnn[i,j])*(phi_yh_s(i,j)-phi_hy_s(i,j))/h
    return -1*(cell_S_x_unn[i,j])*(phi_xh_s(i,j)-phi_hx_s(i,j))/h               #Eq 3.43 (L_phi_star)

def f(cell_cent_phin,i,j):                                                         # Eq 3.54
    if cell_cent_phin[i,j]< -1*M*h:
        return 0
    elif cell_cent_phin[i,j]> M*h:
        return 1
    else:
        return 0.5*(1+(cell_cent_phin[i,j])/(M*h)+(math.sin((math.pi*cell_cent_phin[i,j])/(M*h)))/math.pi)
                                 # Start From Here #Bhsdk
def rho_distr(i,j):
    rho_in=1000
    rho_out=1
    return rho_in*f(i,j)+rho_out*(1-f(i,j))
def mu_distr(i,j):
    mu_in=1e-3
    mu_out=1e-5
    return mu_in*f(i,j)+mu_out*(1-f(i,j))

def us_looper(cell_S_x_un, cell_cent_mu, cell_cent_rho, cell_S_x_us):
    for j in range(1, int(Nx2+1)):
        for i in range(1, int(Nx1+1)):
            cell_S_x_us[i,j]=cell_S_x_un[i,j]+dt*((nu)*Dif_x_n(i,j)+(-gradP)/rho)
    return cell_S_x_us

def p_looper( cell_S_x_us, cell_S_y_vs, cell_cent_pn, cell_cent_pnn):
    global epstot
    epstot=100.0
    while epstot>1e-3:
        epstot=0
        U_s=(rho/(dt*(Lx1/Nx1)))*(cell_S_x_us[2,1]-cell_S_x_us[1,1]+cell_S_y_vs[1,2]-cell_S_y_vs[1,1])
        cell_cent_pnn[1,1]=(cell_vol*U_s-(cell_cent_pn[2,1]+cell_cent_pn[1,2]))/(-2)
        cell_cent_pn[1,1]=cell_cent_pnn[1,1]

        for j in range(2, int(Nx2/2)):
            U_s=(rho/(dt*(Lx1/Nx1)))*(cell_S_x_us[2,j]-cell_S_x_us[1,j]+cell_S_y_vs[1,j+1]-cell_S_y_vs[1,j])
            cell_cent_pnn[1,j]=(cell_vol*U_s-(cell_cent_pn[2,j]+cell_cent_pn[1,j+1]+cell_cent_pn[1,j-1]))/(-3)
            cell_cent_pn[1,j]=cell_cent_pnn[1,j]   
        for i in range(2, int(Nx1)):
            for j in range(1,int(Nx2+1)):
                U_s=(rho/(dt*(Lx1/Nx1)))*(cell_S_x_us[i+1,j]-cell_S_x_us[i,j]+cell_S_y_vs[i,j+1]-cell_S_y_vs[i,j])
                cell_cent_pnn[i,j]=(cell_vol*U_s-(cell_cent_pn[i+1,j]+cell_cent_pn[i-1,j]+cell_cent_pn[i,j+1]+cell_cent_pn[i,j-1]))/(-4)
                cell_cent_pn[i,j]=cell_cent_pnn[i,j]  
                epstot+=(cell_cent_pnn[i,j]-cell_cent_pn[i,j])**2
        for j in range(2, int(Nx2/2)):    
            U_s=(rho/(dt*(Lx1/Nx1)))*(cell_S_x_us[-1,j]-cell_S_x_us[-2,j]+cell_S_y_vs[-2,j+1]-cell_S_y_vs[-2,j])
            cell_cent_pnn[-2,j]=(cell_vol*U_s-(cell_cent_pn[-3,j]+cell_cent_pn[-2,j+1]+cell_cent_pn[-2,j-1]))/(-3)
            cell_cent_pn[-2,j]=cell_cent_pnn[-2,j]

        U_s=(rho/(dt*(Lx1/Nx1)))*(cell_S_x_us[-2+1,1]-cell_S_x_us[-2,1]+cell_S_y_vs[-2,2]-cell_S_y_vnn[-2,1])
        cell_cent_pnn[-2,1]=(cell_vol*U_s-(cell_cent_pn[-2-1,1]+cell_cent_pn[-2,2]))/(-2)
        cell_cent_pn[-2,1]=cell_cent_pnn[-2,1]
        U_s=(rho/(dt*(Lx1/Nx1)))*(cell_S_x_us[2,-2]-cell_S_x_us[1,-2]+cell_S_y_vnn[1,-2]-cell_S_y_vnn[1,-3])
        cell_cent_pnn[1,-2]=(cell_vol*U_s-(cell_cent_pn[2,-2]+cell_cent_pn[1,-3]))/(-2)
        cell_cent_pn[1,-2]=cell_cent_pnn[1,-2]

        for j in range(int(Nx2/2),Nx2):
            U_s=(rho/(dt*(Lx1/Nx1)))*(cell_S_x_us[2,j]-cell_S_x_us[1,j]+cell_S_y_vs[1,j+1]-cell_S_y_vs[1,j])
            cell_cent_pnn[1,j]=(cell_vol*U_s-(cell_cent_pn[2,j]+cell_cent_pn[1,j+1]+cell_cent_pn[1,j-1]))/(-3)
            cell_cent_pn[1,j]=cell_cent_pnn[1,j]   

        for j in range(int(Nx2/2), Nx2):    
            U_s=(rho/(dt*(Lx1/Nx1)))*(cell_S_x_us[-1,j]-cell_S_x_us[-2,j]+cell_S_y_vs[-2,j+1]-cell_S_y_vs[-2,j])
            cell_cent_pnn[-2,j]=(cell_vol*U_s-(cell_cent_pn[-3,j]+cell_cent_pn[-2,j+1]+cell_cent_pn[-2,j-1]))/(-3)
            cell_cent_pn[-2,j]=cell_cent_pnn[-2,j]

        U_s=(rho/(dt*(Lx1/Nx1)))*(cell_S_x_us[-2+1,-2]-cell_S_x_us[-2,-2]+cell_S_y_vnn[-2,-2+1]-cell_S_y_vs[-2,-2])
        cell_cent_pnn[-2,-2]=(cell_vol*U_s-(cell_cent_pn[-2-1,-2]+cell_cent_pn[-2,-2-1]))/(-2)
        cell_cent_pn[-2,-2]=cell_cent_pnn[-2,-2]
    return cell_cent_pn

def p_BC_looper(cell_cent_pn):
    for j in range(0, Nx2+2):
        cell_cent_pn[0,j]=cell_cent_pn[-2,j]     
        cell_cent_pn[-1,j]=cell_cent_pn[1,j]         
    for i in range(0, Nx1+2):
        cell_cent_pn[i,0]=cell_cent_pn[i,1]
        cell_cent_pn[i,-1]=cell_cent_pn[i,-2]
    return cell_cent_pn

def unn_looper(cell_S_x_us, cell_cent_pnn, cell_S_x_unn, cell_cent_rho):
    for j in range(1, int(Nx2+1)):
        for i in range(1, int(Nx1+1)):
            cell_S_x_unn[i,j]=cell_S_x_us[i,j]-(1/cell_cent_rho[i,j])*(dt)*(cell_cent_pnn[i,j]-cell_cent_pnn[i-1,j])
    return cell_S_x_unn
def unn_BC_looper(cell_S_x_unn):
    for j in range(0, Nx2+2):
        cell_S_x_unn[0,j]=cell_S_x_unn[-2,j]     
        cell_S_x_unn[-1,j]=cell_S_x_unn[1,j]         
    for i in range(0, Nx1+2):
        cell_S_x_unn[i,0]=cell_S_x_unn[i,1]
        cell_S_x_unn[i,-1]=cell_S_x_unn[i,-2] 
    return cell_S_x_unn


def phis_looper(cell_cent_phin,cell_cent_phis):
    for j in range(1, int(Nx2+1)):
        for i in range(1, int(Nx1+1)):
            cell_cent_phis[i,j]=cell_cent_phin[i,j]+dt*L_phi_n(i,j)
    return cell_cent_phis

def phis_BC_looper(cell_cent_phis):
    for j in range(0, Nx2+2):
        cell_cent_phis[0,j]=cell_cent_phis[-2,j]       
        cell_cent_phis[-1,j]=cell_cent_phis[1,j]
    for i in range(0, Nx1+2):
        cell_cent_phis[i,0]=cell_cent_phis[i,-2]
        cell_cent_phis[i,-1]=cell_cent_phis[i,1]
    return cell_cent_phis
       
def phinn_looper(cell_cent_phin,cell_cent_phinn):
    for j in range(1, int(Nx2+1)):
        for i in range(1, int(Nx1+1)):
            cell_cent_phinn[i,j]=cell_cent_phin[i,j]+0.5*dt*(L_phi_n(i,j)+L_phi_s(i,j))
    return cell_cent_phinn

def phinn_BC_looper(cell_cent_phinn):
    for j in range(0, Nx2+2):
        cell_cent_phinn[0,j]=cell_cent_phinn[-2,j]
        cell_cent_phinn[-1,j]=cell_cent_phinn[1,j]
        
    for i in range(0, Nx1+2):
        cell_cent_phinn[i,0]=cell_cent_phinn[i,-2]
        cell_cent_phinn[i,-1]=cell_cent_phinn[i,1]
    return cell_cent_phinn
