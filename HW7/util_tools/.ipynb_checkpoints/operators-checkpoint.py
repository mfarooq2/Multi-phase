
# def Adv_x_n(i,j):
#     return (1/h)*((0.5*(cell_S_x_un[i,j]+cell_S_x_un[i+1,j]))**2-(0.5*(cell_S_x_un[i,j]+cell_S_x_un[i-1,j]))**2+(0.5*(cell_S_x_un[i,j+1]+cell_S_x_un[i,j]))*(cell_S_y_vn[i,j+1]+cell_S_y_vn[i+1,j+1])-(0.5*(cell_S_x_un[i,j]+cell_S_x_un[i,j-1]))*(0.5*(cell_S_y_vn[i,j]+cell_S_y_vn[i+1,j]))) 
import math
import numpy as np
def Dif_x_n(cell_S_x_un,var_dict):
    return (
        1
        / (var_dict['h']**2)
        * (
            cell_S_x_un[1 + 1 : var_dict['Nx1'] + 1 + 1, 1 : var_dict['Nx2'] + 1]
            + cell_S_x_un[0 : var_dict['Nx1'] + 1 - 1, 1 : var_dict['Nx2'] + 1]
            + cell_S_x_un[0 : var_dict['Nx1'] + 1 - 1, 1 + 1 : var_dict['Nx2'] + 1 + 1]
            + cell_S_x_un[0 : var_dict['Nx1'] + 1 - 1, 0 : var_dict['Nx2'] + 1 - 1]
            - 4 * cell_S_x_un[0 : var_dict['Nx1'] + 1 - 1, 1 : var_dict['Nx2'] + 1]
        )
    )
#def Adv_y_n(i,j):
#    return (1/h)*((0.5*())**2-(0.5*())**2+(0.5*())*(0.5*())-(0.5*())*(0.5*()))

def Dif_y_n(cell_S_y_vn,var_dict):
    return (
        1
        / (var_dict['h']**2)
        * (
            cell_S_y_vn[1 + 1 : var_dict['Nx1'] + 1 + 1, 1 : var_dict['Nx2'] + 1]
            + cell_S_y_vn[0 : var_dict['Nx1'] + 1 - 1, 1 : var_dict['Nx2'] + 1]
            + cell_S_y_vn[1 : var_dict['Nx1'] + 1, 1 + 1 : var_dict['Nx2'] + 1 + 1]
            + cell_S_y_vn[1 : var_dict['Nx1'] + 1, 0 : var_dict['Nx2'] + 1 - 1]
            - 4 * cell_S_y_vn[1 : var_dict['Nx1'] + 1, 1 : var_dict['Nx2'] + 1]
        )
    )

def ref_vel_prof(x2):
    return -1200*((x2-0.005)**2)+0.03
    

def lvlset_init(x,y,var_dict):
    def ls_dpl(x,y):
        return -1*(math.sqrt((x-0.01)**2+(y-0.005)**2)-var_dict['r_dpl'])
    return ls_dpl(x,y)

def M_sw(a,b):
    return a if abs(a)<abs(b) else b
#level-set functions

def D_x_p_n(cell_cent_phin,i,j,var_dict):
    if i==var_dict['Nx1']+1:
        return cell_cent_phin[2,j]-cell_cent_phin[i,j]
    else:    
        return cell_cent_phin[i+1,j]-cell_cent_phin[i,j]

def D_x_m_n(cell_cent_phin,i,j):
    return cell_cent_phin[i,j]-cell_cent_phin[i-1,j]

def D_y_p_n(cell_cent_phin,i,j):
    return cell_cent_phin[i,j+1]-cell_cent_phin[i,j]

def D_y_m_n(cell_cent_phin,i,j):
    return cell_cent_phin[i,j]-cell_cent_phin[i-1,j]

def D_x_p_s(cell_cent_phis,i,j,var_dict):
    if i==var_dict['Nx1']+1:
        return cell_cent_phis[2,j]-cell_cent_phis[i,j]
    else:    
        return cell_cent_phis[i+1,j]-cell_cent_phis[i,j]                     

def D_x_m_s(cell_cent_phis,i,j):       
    return cell_cent_phis[i,j]-cell_cent_phis[i-1,j]                      

def D_y_p_s(cell_cent_phis,i,j):       
    return cell_cent_phis[i,j+1]-cell_cent_phis[i,j]                   

def D_y_m_s(cell_cent_phis,i,j):       
    return cell_cent_phis[i,j]-cell_cent_phis[i-1,j]

def L_phi_n(cell_cent_phin,cell_S_x_unn,cell_S_y_vnn,var_dict):
    #cell_cent_phin[1:Nx1+1,1:Nx2+1]
    h=var_dict['h']
    def phi_xh_n(i,j):
        if 0.5*(cell_S_x_unn[i+1,j]+cell_S_x_unn[i,j])>0:
            return cell_cent_phin[i,j]+0.5*M_sw(D_x_p_n(i,j), D_x_m_n(i,j))
        elif 0.5*(cell_S_x_unn[i+1,j]+cell_S_x_unn[i,j])<0:
            return cell_cent_phin[i+1,j]-0.5*M_sw(D_x_p_n(i+1,j), D_x_m_n(i+1,j))
                
    def phi_hx_n(i,j):
        if 0.5*(cell_S_x_unn[i,j]+cell_S_x_unn[i-1,j])<0:
            return cell_cent_phin[i,j]-0.5*M_sw(D_x_p_n(i,j), D_x_m_n(i,j)) 
        elif 0.5*(cell_S_x_unn[i,j]+cell_S_x_unn[i-1,j])>0:
            return cell_cent_phin[i-1,j]+0.5*M_sw(D_x_p_n(i-1,j), D_x_m_n(i-1,j))
    def phi_yh_n(i,j):
        if 0.5*(cell_S_y_vnn[i,j+1]+cell_S_y_vnn[i,j])>0:
            return cell_cent_phin[i,j]+0.5*M_sw(D_y_p_n(i,j), D_y_m_n(i,j))
        elif 0.5*(cell_S_y_vnn[i,j+1]+cell_S_y_vnn[i,j])<0:
            return cell_cent_phin[i,j+1]+0.5*M_sw(D_y_p_n(i,j+1), D_y_m_n(i,j+1))
    def phi_hy_n(i,j):
        if 0.5*(cell_S_y_vnn[i,j]+cell_S_y_vnn[i,j-1])<0:
            return cell_cent_phin[i,j]-0.5*M_sw(D_y_p_n(i,j), D_y_m_n(i,j))
        elif 0.5*(cell_S_y_vnn[i,j]+cell_S_y_vnn[i,j-1])>0:
            return cell_cent_phin[i,j-1]+0.5*M_sw(D_y_p_n(i,j-1), D_y_m_n(i,j-1))
   
    #return -1*(cell_S_x_unn[i,j])*(phi_xh_n(i,j)-phi_hx_n(i,j))/h-(cell_S_y_vnn[i,j])*(phi_yh_n(i,j)-phi_hy_n(i,j))/h
    return -1*(cell_S_x_unn[i,j])*(phi_xh_n(i,j)-phi_hx_n(i,j))/h

def L_phi_s(i,j,cell_S_x_unn,cell_S_y_vnn,cell_cent_phis,var_dict):
    h=var_dict['h']
    def phi_xh_s(i,j):
        if 0.5*(cell_S_x_unn[i+1,j]+cell_S_x_unn[i,j])>0:
            return cell_cent_phis[i,j]+0.5*M_sw(D_x_p_s(i,j), D_x_m_s(i,j))
        elif 0.5*(cell_S_x_unn[i+1,j]+cell_S_x_unn[i,j])<0:
            return cell_cent_phis[i+1,j]-0.5*M_sw(D_x_p_s(i+1,j), D_x_p_s(i+1,j))
            
    def phi_hx_s(i,j):
        if 0.5*(cell_S_x_unn[i,j]+cell_S_x_unn[i-1,j])<0:
            return cell_cent_phis[i,j]-0.5*M_sw(D_x_p_s(i,j), D_x_m_s(i,j)) 
        elif 0.5*(cell_S_x_unn[i,j]+cell_S_x_unn[i-1,j])>0:
            return cell_cent_phis[i-1,j]+0.5*M_sw(D_x_p_s(i-1,j), D_x_m_s(i-1,j))
    def phi_yh_s(i,j):
        if 0.5*(cell_S_y_vnn[i,j+1]+cell_S_y_vnn[i,j])>0:
            return cell_cent_phis[i,j]+0.5*M_sw(D_y_p_s(i,j), D_y_m_s(i,j))
        elif 0.5*(cell_S_y_vnn[i,j+1]+cell_S_y_vnn[i,j])<0:
            return cell_cent_phis[i,j+1]+0.5*M_sw(D_y_p_s(i,j+1), D_y_m_s(i,j+1))
    def phi_hy_s(i,j):
        if 0.5*(cell_S_y_vnn[i,j]+cell_S_y_vnn[i,j-1])<0:
            return cell_cent_phis[i,j]-0.5*M_sw(D_y_p_s(i,j), D_y_m_s(i,j))
        elif 0.5*(cell_S_y_vnn[i,j]+cell_S_y_vnn[i,j-1])>0:
            return cell_cent_phis[i,j-1]+0.5*M_sw(D_y_p_s(i,j-1), D_y_m_s(i,j-1))
    return -1*(cell_S_x_unn[i,j])*(phi_xh_s(i,j)-phi_hx_s(i,j))/h

#redistancing functions

def sign_phi_Mh(cell_cent_phin,i,j,var_dict):
    M=var_dict['M']
    h=var_dict['h']
    if cell_cent_phin[i,j]>=(M*h):
        return 1
    elif cell_cent_phin[i,j]<=(-1*M*h):
        return -1
    else:
        return cell_cent_phin[i,j]/(M*h)-math.sin((math.pi*cell_cent_phin[i,j])/(M*h))/math.pi

def sign_phi(cell_cent_phin,i,j):
    if cell_cent_phin[i,j]>0:
        return 1
    elif cell_cent_phin[i,j]<0:
        return -1
    else:
        return 0
    
   
def Dd_x_p_n(cell_cent_phi_dn,i,j):
    return cell_cent_phi_dn[i+1,j]-cell_cent_phi_dn[i,j]

def Dd_x_m_n(cell_cent_phi_dn,i,j):
    return cell_cent_phi_dn[i,j]-cell_cent_phi_dn[i-1,j]

def Dd_y_p_n(cell_cent_phi_dn,i,j):
    return cell_cent_phi_dn[i,j+1]-cell_cent_phi_dn[i,j]

def Dd_y_m_n(cell_cent_phi_dn,i,j):
    return cell_cent_phi_dn[i,j]-cell_cent_phi_dn[i,j-1]


def DDd_pm_x_n(cell_cent_phi_dn,i,j,var_dict):
    if i==var_dict['Nx1']+1:
        return cell_cent_phi_dn[2,j]-2*cell_cent_phi_dn[i,j]+cell_cent_phi_dn[i-1,j]
    elif i==0:
        return cell_cent_phi_dn[i+1,j]-2*cell_cent_phi_dn[i,j]+cell_cent_phi_dn[-3,j]
    else:
        return cell_cent_phi_dn[i+1,j]-2*cell_cent_phi_dn[i,j]+cell_cent_phi_dn[i-1,j]

def DDd_pm_y_n(cell_cent_phi_dn,i,j,var_dict):
    if j==var_dict['Nx2']+1:
        return cell_cent_phi_dn[i,2]-2*cell_cent_phi_dn[i,1]+cell_cent_phi_dn[i,0]
    elif j==0:
        return cell_cent_phi_dn[i,-1]-2*cell_cent_phi_dn[i,-2]+cell_cent_phi_dn[i,-3]
    else:
        return cell_cent_phi_dn[i,j+1]-2*cell_cent_phi_dn[i,j]+cell_cent_phi_dn[i,j-1]



def Dtda_x_p_n(i,j):
    return Dd_x_p_n(i,j)-0.5*M_sw(DDd_pm_x_n(i,j), DDd_pm_x_n(i+1,j)) 

def Dtda_x_m_n(i,j):
    return Dd_x_m_n(i,j)+0.5*M_sw(DDd_pm_x_n(i,j), DDd_pm_x_n(i-1,j))

def Dtda_y_p_n(i,j):
    return Dd_y_p_n(i,j)-0.5*M_sw(DDd_pm_y_n(i,j), DDd_pm_y_n(i+1,j))

def Dtda_y_m_n(i,j):
    return Dd_y_m_n(i,j)+0.5*M_sw(DDd_pm_y_n(i,j), DDd_pm_y_n(i-1,j))


def Dtda_x_n(i,j):
    if (sign_phi(i,j)*Dd_x_p_n(i,j)<0) and (sign_phi(i,j)*Dd_x_m_n(i,j)<-1*sign_phi(i,j)*Dd_x_p_n(i,j)):
        return Dtda_x_p_n(i,j)
    elif(sign_phi(i,j)*Dd_x_m_n(i,j)<0) and (sign_phi(i,j)*Dd_x_p_n(i,j)>-1*sign_phi(i,j)*Dd_x_m_n(i,j)):
        return Dtda_x_m_n(i,j)
    else:
        return 0.5*(Dtda_x_p_n(i,j)+Dtda_x_m_n(i,j))  

def Dtda_y_n(i,j):
    if (sign_phi(i,j)*Dd_y_p_n(i,j)<0) and (sign_phi(i,j)*Dd_y_m_n(i,j)<-1*sign_phi(i,j)*Dd_y_p_n(i,j)):
        return Dtda_y_p_n(i,j)
    elif (sign_phi(i,j)*Dd_y_m_n(i,j)<0) and (sign_phi(i,j)*Dd_y_p_n(i,j)>-1*sign_phi(i,j)*Dd_y_m_n(i,j)):
        return Dtda_y_m_n(i,j)
    else:
        return 0.5*(Dtda_y_p_n(i,j)+Dtda_y_m_n(i,j))
    
def L_phi_d_n(i,j,var_dict):
    h=var_dict['h']
    return sign_phi_Mh(i,j)*(1-((Dtda_x_n(i,j)/h)**2+(Dtda_y_n(i,j)/h)**2)**0.5)



def Dd_x_p_s(cell_cent_phi_ds):
    return cell_cent_phi_ds[i+1,j]-cell_cent_phi_ds[i,j]

def Dd_x_m_s(cell_cent_phi_ds,i,j):
    return cell_cent_phi_ds[i,j]-cell_cent_phi_ds[i-1,j]

def Dd_y_p_s(cell_cent_phi_ds):
    return cell_cent_phi_ds[i,j+1]-cell_cent_phi_ds[i,j]

def Dd_y_m_s(cell_cent_phi_ds,i,j):
    return cell_cent_phi_ds[i,j]-cell_cent_phi_ds[i,j-1]

def DDd_pm_x_s(cell_cent_phi_ds,i,j,var_dict):
    if i==var_dict['Nx1']+1:
        return cell_cent_phi_ds[2,j]-2*cell_cent_phi_ds[i,j]+cell_cent_phi_ds[i-1,j]
    elif i==0:
        return cell_cent_phi_ds[i+1,j]-2*cell_cent_phi_ds[i,j]+cell_cent_phi_ds[-3,j]
    else:
        return cell_cent_phi_ds[i+1,j]-2*cell_cent_phi_ds[i,j]+cell_cent_phi_ds[i-1,j]

def DDd_pm_y_s(cell_cent_phi_ds,i,j,var_dict):
    if i==var_dict['Nx1']+1:
        return cell_cent_phi_ds[2,j]-2*cell_cent_phi_ds[i,j]+cell_cent_phi_ds[i-1,j]
    elif i==0:
        return cell_cent_phi_ds[i+1,j]-2*cell_cent_phi_ds[i,j]+cell_cent_phi_ds[-3,j]
    else:
        return cell_cent_phi_ds[i,j+1]-2*cell_cent_phi_ds[i,j]+cell_cent_phi_ds[i,j-1]

        

def Dtda_x_p_s(i,j):
    return Dd_x_p_s(i,j)-0.5*M_sw(DDd_pm_x_s(i,j), DDd_pm_x_s(i+1,j)) 

def Dtda_x_m_s(i,j):
    return Dd_x_m_s(i,j)+0.5*M_sw(DDd_pm_x_s(i,j), DDd_pm_x_s(i-1,j))

def Dtda_y_p_s(i,j):
    return Dd_y_p_s(i,j)-0.5*M_sw(DDd_pm_y_s(i,j), DDd_pm_y_s(i+1,j))

def Dtda_y_m_s(i,j):
    return Dd_y_m_s(i,j)+0.5*M_sw(DDd_pm_y_s(i,j), DDd_pm_y_s(i-1,j))


def Dtda_x_s(i,j):
    if (sign_phi(i,j)*Dd_x_p_s(i,j)<0) and (sign_phi(i,j)*Dd_x_m_s(i,j)<-1*sign_phi(i,j)*Dd_x_p_s(i,j)):
        return Dtda_x_p_s(i,j)
    elif(sign_phi(i,j)*Dd_x_m_s(i,j)<0) and (sign_phi(i,j)*Dd_x_p_s(i,j)>-1*sign_phi(i,j)*Dd_x_m_s(i,j)):
        return Dtda_x_m_s(i,j)
    else:
        return 0.5*(Dtda_x_p_s(i,j)+Dtda_x_m_s(i,j))  

def Dtda_y_s(i,j):
    if (sign_phi(i,j)*Dd_y_p_s(i,j)<0) and (sign_phi(i,j)*Dd_y_m_s(i,j)<-1*sign_phi(i,j)*Dd_y_p_s(i,j)):
        return Dtda_y_p_s(i,j)
    elif (sign_phi(i,j)*Dd_y_m_s(i,j)<0) and (sign_phi(i,j)*Dd_y_p_s(i,j)>-1*sign_phi(i,j)*Dd_y_m_s(i,j)):
        return Dtda_y_m_s(i,j)
    else:
        return 0.5*(Dtda_y_p_s(i,j)+Dtda_y_m_s(i,j))
    
    
def L_phi_d_s(i,j):
    return sign_phi_Mh(i,j)*(1-((Dtda_x_s(i,j)/h)**2+(Dtda_y_s(i,j)/h)**2)**0.5)    


def grad_phi_xhpp(cell_cent_phin,i,j):
    return (cell_cent_phin[i+1,j+1]+cell_cent_phin[i+1,j]-cell_cent_phin[i,j+1]-cell_cent_phin[i,j])/(2*h)

def grad_phi_xhpm(cell_cent_phin,i,j):
    return (cell_cent_phin[i+1,j]+cell_cent_phin[i+1,j-1]-cell_cent_phin[i,j]-cell_cent_phin[i,j-1])/(2*h)

def grad_phi_xhmp(cell_cent_phin):
    return (cell_cent_phin[i,j+1]+cell_cent_phin[i,j]-cell_cent_phin[i-1,j+1]-cell_cent_phin[i-1,j])/(2*h)

def grad_phi_xhmm(cell_cent_phin):
    return (cell_cent_phin[i,j]+cell_cent_phin[i,j-1]-cell_cent_phin[i-1,j]-cell_cent_phin[i-1,j-1])/(2*h)


def grad_phi_yhpp(cell_cent_phin):
    return (cell_cent_phin[i+1,j+1]-cell_cent_phin[i+1,j]+cell_cent_phin[i,j+1]-cell_cent_phin[i,j])/(2*h)

def grad_phi_yhpm(cell_cent_phin):
    return (cell_cent_phin[i+1,j]-cell_cent_phin[i+1,j-1]+cell_cent_phin[i,j]-cell_cent_phin[i,j-1])/(2*h)

def grad_phi_yhmp(cell_cent_phin):
    return (cell_cent_phin[i,j+1]-cell_cent_phin[i,j]+cell_cent_phin[i-1,j+1]-cell_cent_phin[i-1,j])/(2*h)

def grad_phi_yhmm(cell_cent_phin):
    return (cell_cent_phin[i,j]-cell_cent_phin[i,j-1]+cell_cent_phin[i-1,j]+cell_cent_phin[i-1,j-1])/(2*h)

def grad_phi_mag_hpp(grad_phi_xhpp):
    return (grad_phi_xhpp(i,j)**2+grad_phi_yhpp(i,j)**2)**0.5

def grad_phi_mag_hpm(grad_phi_xhpm):
    return (grad_phi_xhpm(i,j)**2+grad_phi_yhpm(i,j)**2)**0.5

def grad_phi_mag_hmp(i,j):
    return (grad_phi_xhmp(i,j)**2+grad_phi_yhmp(i,j)**2)**0.5

def grad_phi_mag_hmm(i,j):
    return (grad_phi_xhmm(i,j)**2+grad_phi_yhmm(i,j)**2)**0.5

def kappa(i,j):
    return ((grad_phi_xhpp(i,j)/grad_phi_mag_hpp(i,j))+(grad_phi_xhpm(i,j)/grad_phi_mag_hpm(i,j))-(grad_phi_xhmp(i,j)/grad_phi_mag_hmp(i,j))-(grad_phi_xhmm(i,j)/grad_phi_mag_hmm(i,j))+(grad_phi_yhpp(i,j)/grad_phi_mag_hpp(i,j))-(grad_phi_yhpm(i,j)/grad_phi_mag_hpm(i,j))+(grad_phi_yhmp(i,j)/grad_phi_mag_hmp(i,j))-(grad_phi_yhmm(i,j)/grad_phi_mag_hmm(i,j)))


def dfdphi(cell_cent_phin,var_dict):
    M=var_dict['M']
    h=var_dict['h']
    return np.where(cell_cent_phin<-1*M*h or cell_cent_phin>1*M*h,0,(1+math.cos((math.pi*cell_cent_phin)/(M*h)))/(2*M*h))
    
def grad_phi_x(cell_cent_phin,i,j,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    return (cell_cent_phin[1+1:Nx1+1+1,1: Nx2+1]-cell_cent_phin[1-1:Nx1+1-1,1: Nx2+1])/(2*var_dict['h'])



def f_st_x(cell_cent_phin,var_dict):
    return var_dict['st_coef']*kappa(i,j)*dfdphi(cell_cent_phin,var_dict)*grad_phi_x(i,j,var_dict)


def f(cell_cent_phin,var_dict):
    h=var_dict['h']
    M=var_dict['M']
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    return np.where(cell_cent_phin[0:Nx1+2,0:Nx2+2]<-1*M*h,0,np.where(cell_cent_phin[0:Nx1+2,0:Nx2+2]>M*h,1,0.5*(1+(cell_cent_phin[0:Nx1+2,0:Nx2+2])/(M*h)+(math.sin((math.pi*cell_cent_phin)/(M*h)))/math.pi)))
def rho_distr(cell_cent_phin,var_dict):
    rho_in=1000
    rho_out=999
    return rho_in*f(cell_cent_phin,var_dict)+rho_out*(1-f(cell_cent_phin,var_dict))
def mu_distr(cell_cent_phin,var_dict):
    mu_in=1e-3
    mu_out=1e-4
    return mu_in**f(cell_cent_phin,var_dict)+mu_out*(1-f(cell_cent_phin,var_dict))