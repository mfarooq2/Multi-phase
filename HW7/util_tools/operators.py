
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
    return -1*(np.sqrt((x-0.01)**2+(y-0.005)**2)-var_dict['r_dpl'])

#level-set functions
def L_phi_n(cell_cent_phis,cell_cent_phin,cell_S_x_un,cell_S_y_vn,var_dict):
    #cell_cent_phin[1:Nx1+1,1:Nx2+1]
    
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    h=var_dict['h']
    D_x_p_n=np.zeros_like(cell_cent_phin)
    D_x_m_n=np.zeros_like(cell_cent_phin)
    D_y_p_n=np.zeros_like(cell_cent_phin)
    D_y_m_n=np.zeros_like(cell_cent_phin)
    D_x_p_s=np.zeros_like(cell_cent_phis)
    D_x_m_s=np.zeros_like(cell_cent_phis)
    D_y_p_s=np.zeros_like(cell_cent_phis)
    D_y_m_s=np.zeros_like(cell_cent_phis)
    D_x_p_s=np.zeros_like(cell_cent_phis)
    phi_xh_n=np.zeros_like(cell_S_x_un)
    phi_hx_n=np.zeros_like(cell_S_x_un)

    D_x_p_n[Nx1+1,1:Nx2+1] = cell_cent_phin[2,1:Nx2+1]-cell_cent_phin[Nx1+1,1:Nx2+1]
    D_x_p_n[1:Nx1+1,1:Nx2+1]= cell_cent_phin[1+1:Nx1+1+1,1:Nx2+1]-cell_cent_phin[1:Nx1+1,1:Nx2+1]
    D_x_m_n[1:Nx1+1,1:Nx2+1]= cell_cent_phin[1:Nx1+1,1:Nx2+1]-cell_cent_phin[1-1:Nx1+1-1,1:Nx2+1]
    D_y_p_n[1:Nx1+1,1:Nx2+1]= cell_cent_phin[1:Nx1+1,1+1:Nx2+1+1]-cell_cent_phin[1:Nx1+1,1:Nx2+1]
    D_y_m_n[1:Nx1+1,1:Nx2+1]= cell_cent_phin[1:Nx1+1,1:Nx2+1]-cell_cent_phin[1:Nx1+1,1-1:Nx2+1-1]
    D_x_p_s[Nx1+1,1:Nx2+1] = cell_cent_phis[2,1:Nx2+1]-cell_cent_phis[Nx1+1,1:Nx2+1] 
    D_x_p_s[1:Nx1+1,1:Nx2+1]= cell_cent_phis[1+1:Nx1+1+1,1:Nx2+1]-cell_cent_phis[1:Nx1+1,1:Nx2+1]                     
    D_x_m_s[1:Nx1+1,1:Nx2+1]= cell_cent_phis[1:Nx1+1,1:Nx2+1]-cell_cent_phis[1-1:Nx1+1-1,1:Nx2+1]                      
    D_y_p_s[1:Nx1+1,1:Nx2+1]= cell_cent_phis[1:Nx1+1,1+1:Nx2+1+1]-cell_cent_phis[1:Nx1+1,1:Nx2+1]                   
    D_y_m_s[1:Nx1+1,1:Nx2+1]= cell_cent_phis[1:Nx1+1,1:Nx2+1]-cell_cent_phis[1-1:Nx1+1-1,1:Nx2+1]

    phi_xh_n[1:Nx1+1,1:Nx2+1]=np.where(0.5*(cell_S_x_un[1+1:Nx1+1+1,1:Nx2+1]+cell_S_x_un[1:Nx1+1,1:Nx2+1])>0,cell_cent_phin[1:Nx1+1,1:Nx2+1] + 0.5*np.where((abs(D_x_p_n[1:Nx1+1,1:Nx2+1])<abs(D_x_m_n[1:Nx1+1,1:Nx2+1])),D_x_p_n[1:Nx1+1,1:Nx2+1],D_x_m_n[1:Nx1+1,1:Nx2+1]),cell_cent_phin[1+1:Nx1+1+1,1:Nx2+1] - 0.5*np.where((abs(D_x_p_n[1+1:Nx1+1+1,1:Nx2+1])<abs(D_x_m_n[1+1:Nx1+1+1,1:Nx2+1])),D_x_m_n[1+1:Nx1+1+1,1:Nx2+1],D_x_m_n[1+1:Nx1+1+1,1:Nx2+1]))
    phi_hx_n[1:Nx1+1,1:Nx2+1]=np.where(0.5*(cell_S_x_un[1:Nx1+1,1:Nx2+1]+cell_S_x_un[1-1:Nx1+1-1,1:Nx2+1])>0,cell_cent_phin[1:Nx1+1,1:Nx2+1] + 0.5*np.where((abs(D_x_p_n[1-1:Nx1+1-1,1:Nx2+1])<abs(D_x_m_n[1-1:Nx1+1-1,1:Nx2+1])),D_x_p_n[1-1:Nx1+1-1,1:Nx2+1],D_x_m_n[1-1:Nx1+1-1,1:Nx2+1]),cell_cent_phin[1:Nx1+1,1:Nx2+1] - 0.5*np.where((abs(D_x_p_n[1:Nx1+1,1:Nx2+1])<abs(D_x_m_n[1:Nx1+1,1:Nx2+1])),D_x_p_n[1:Nx1+1,1:Nx2+1],D_x_m_n[1:Nx1+1,1:Nx2+1]))


    return -1*(cell_S_x_un)*(phi_xh_n-phi_hx_n)/h



def L_phi_s(cell_cent_phis,cell_cent_phin,cell_S_x_us,cell_S_y_vn,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    h=var_dict['h']
    D_x_p_n=np.zeros_like(cell_cent_phin)
    D_x_m_n=np.zeros_like(cell_cent_phin)
    D_y_p_n=np.zeros_like(cell_cent_phin)
    D_y_m_n=np.zeros_like(cell_cent_phin)
    D_x_p_s=np.zeros_like(cell_cent_phis)
    D_x_m_s=np.zeros_like(cell_cent_phis)
    D_y_p_s=np.zeros_like(cell_cent_phis)
    D_y_m_s=np.zeros_like(cell_cent_phis)
    D_x_p_s=np.zeros_like(cell_cent_phis)
    phi_xh_s=np.zeros_like(cell_S_x_us)
    phi_hx_s=np.zeros_like(cell_S_x_us)

    D_x_p_n[Nx1+1,1:Nx2+1] = cell_cent_phin[2,1:Nx2+1]-cell_cent_phin[Nx1+1,1:Nx2+1]
    D_x_p_n[1:Nx1+1,1:Nx2+1]= cell_cent_phin[1+1:Nx1+1+1,1:Nx2+1]-cell_cent_phin[1:Nx1+1,1:Nx2+1]
    D_x_m_n[1:Nx1+1,1:Nx2+1]= cell_cent_phin[1:Nx1+1,1:Nx2+1]-cell_cent_phin[1-1:Nx1+1-1,1:Nx2+1]
    D_y_p_n[1:Nx1+1,1:Nx2+1]= cell_cent_phin[1:Nx1+1,1+1:Nx2+1+1]-cell_cent_phin[1:Nx1+1,1:Nx2+1]
    D_y_m_n[1:Nx1+1,1:Nx2+1]= cell_cent_phin[1:Nx1+1,1:Nx2+1]-cell_cent_phin[1:Nx1+1,1-1:Nx2+1-1]
    D_x_p_s[Nx1+1,1:Nx2+1] = cell_cent_phis[2,1:Nx2+1]-cell_cent_phis[Nx1+1,1:Nx2+1] 
    D_x_p_s[1:Nx1+1,1:Nx2+1]= cell_cent_phis[1+1:Nx1+1+1,1:Nx2+1]-cell_cent_phis[1:Nx1+1,1:Nx2+1]                     
    D_x_m_s[1:Nx1+1,1:Nx2+1]= cell_cent_phis[1:Nx1+1,1:Nx2+1]-cell_cent_phis[1-1:Nx1+1-1,1:Nx2+1]                      
    D_y_p_s[1:Nx1+1,1:Nx2+1]= cell_cent_phis[1:Nx1+1,1+1:Nx2+1+1]-cell_cent_phis[1:Nx1+1,1:Nx2+1]                   
    D_y_m_s[1:Nx1+1,1:Nx2+1]= cell_cent_phis[1:Nx1+1,1:Nx2+1]-cell_cent_phis[1-1:Nx1+1-1,1:Nx2+1]

    phi_xh_s[1:Nx1+1,1:Nx2+1]=np.where(0.5*(cell_S_x_us[1+1:Nx1+1+1,1:Nx2+1]+cell_S_x_us[1:Nx1+1,1:Nx2+1])>0,cell_cent_phis[1:Nx1+1,1:Nx2+1] + 0.5*np.where((abs(D_x_p_s[1:Nx1+1,1:Nx2+1])<abs(D_x_m_s[1:Nx1+1,1:Nx2+1])),D_x_p_s[1:Nx1+1,1:Nx2+1],D_x_m_s[1:Nx1+1,1:Nx2+1]),cell_cent_phis[1+1:Nx1+1+1,1:Nx2+1] - 0.5*np.where((abs(D_x_p_s[1+1:Nx1+1+1,1:Nx2+1])<abs(D_x_m_s[1+1:Nx1+1+1,1:Nx2+1])),D_x_m_s[1+1:Nx1+1+1,1:Nx2+1],D_x_m_s[1+1:Nx1+1+1,1:Nx2+1]))
    phi_hx_s[1:Nx1+1,1:Nx2+1]=np.where(0.5*(cell_S_x_us[1:Nx1+1,1:Nx2+1]+cell_S_x_us[1-1:Nx1+1-1,1:Nx2+1])>0,cell_cent_phis[1:Nx1+1,1:Nx2+1] + 0.5*np.where((abs(D_x_p_s[1-1:Nx1+1-1,1:Nx2+1])<abs(D_x_m_s[1-1:Nx1+1-1,1:Nx2+1])),D_x_p_s[1-1:Nx1+1-1,1:Nx2+1],D_x_m_s[1-1:Nx1+1-1,1:Nx2+1]),cell_cent_phis[1:Nx1+1,1:Nx2+1] - 0.5*np.where((abs(D_x_p_s[1:Nx1+1,1:Nx2+1])<abs(D_x_m_s[1:Nx1+1,1:Nx2+1])),D_x_p_s[1:Nx1+1,1:Nx2+1],D_x_m_s[1:Nx1+1,1:Nx2+1]))


    return -1*(cell_S_x_us)*(phi_xh_s-phi_hx_s)/h


def L_phi_d_n(cell_cent_phin,cell_cent_phi_dn,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    h=var_dict['h']
    M=var_dict['M']
    sign_phi=np.zeros_like(cell_cent_phin)
    sign_phi_Mh=np.zeros_like(cell_cent_phin)
    Dd_x_p_n=np.zeros_like(cell_cent_phi_dn)
    Dd_x_m_n=np.zeros_like(cell_cent_phi_dn)
    Dd_y_p_n=np.zeros_like(cell_cent_phi_dn)
    Dd_y_m_n=np.zeros_like(cell_cent_phi_dn)
    DDd_pm_x_n=np.zeros_like(cell_cent_phi_dn)
    DDd_pm_y_n=np.zeros_like(cell_cent_phi_dn)
    
    Dtda_x_p_n=np.zeros_like(Dd_x_p_n)
    Dtda_x_m_n=np.zeros_like(Dd_y_m_n)
    Dtda_y_p_n=np.zeros_like(Dd_y_p_n)
    Dtda_y_m_n=np.zeros_like(Dd_y_m_n)
    Dtda_x_n=np.zeros_like(Dd_x_m_n)
    Dtda_y_n=np.zeros_like(Dd_y_p_n)
    
    
    
    sign_phi[1:Nx1+1,1:Nx2+1]=np.where(cell_cent_phin[1:Nx1+1,1:Nx2+1]>0,1,np.where(cell_cent_phin[1:Nx1+1,1:Nx2+1]<0,-1,0))
    sign_phi_Mh[1:Nx1+1,1:Nx2+1]=np.where(cell_cent_phin[1:Nx1+1,1:Nx2+1]>(M*h),1,np.where(cell_cent_phin[1:Nx1+1,1:Nx2+1]<(-1*M*h),-1,cell_cent_phin[1:Nx1+1,1:Nx2+1]/(M*h)-np.sin((np.pi*cell_cent_phin[1:Nx1+1,1:Nx2+1])/(M*h))/np.pi))


    Dd_x_p_n[1:Nx1+1,1:Nx2+1]= cell_cent_phi_dn[1+1:Nx1+1+1,1:Nx2+1]-cell_cent_phi_dn[1:Nx1+1,1:Nx2+1]
    Dd_x_m_n[1:Nx1+1,1:Nx2+1]= cell_cent_phi_dn[1:Nx1+1,1:Nx2+1]-cell_cent_phi_dn[1-1:Nx1+1-1,1:Nx2+1]
    Dd_y_p_n[1:Nx1+1,1:Nx2+1]= cell_cent_phi_dn[1:Nx1+1,1+1:Nx2+1+1]-cell_cent_phi_dn[1:Nx1+1,1:Nx2+1]
    Dd_y_m_n[1:Nx1+1,1:Nx2+1]= cell_cent_phi_dn[1:Nx1+1,1:Nx2+1]-cell_cent_phi_dn[1:Nx1+1,1-1:Nx2+1-1]
    DDd_pm_x_n[Nx1+1,1:Nx2+1] =  cell_cent_phi_dn[2,1:Nx2+1]-2*cell_cent_phi_dn[Nx1+1,1:Nx2+1]+cell_cent_phi_dn[Nx1+1-1,1:Nx2+1]
    DDd_pm_x_n[0,1:Nx2+1]= cell_cent_phi_dn[0+1,1:Nx2+1]-2*cell_cent_phi_dn[0,1:Nx2+1]+cell_cent_phi_dn[-3,1:Nx2+1]
    DDd_pm_x_n[1:Nx1+1,1:Nx2+1]= cell_cent_phi_dn[1+1:Nx1+1+1,1:Nx2+1]-2*cell_cent_phi_dn[1:Nx1+1,1:Nx2+1]+cell_cent_phi_dn[1-1:Nx1+1-1,1:Nx2+1]
    DDd_pm_y_n[1:Nx1+1,Nx2+1] =  cell_cent_phi_dn[1:Nx1+1,2]-2*cell_cent_phi_dn[1:Nx1+1,1]+cell_cent_phi_dn[1:Nx1+1,0]
    DDd_pm_y_n[1:Nx1+1,0]= cell_cent_phi_dn[1:Nx1+1,-1]-2*cell_cent_phi_dn[1:Nx1+1,-2]+cell_cent_phi_dn[1:Nx1+1,-3]
    DDd_pm_y_n[1:Nx1+1,1:Nx2+1]= cell_cent_phi_dn[1:Nx1+1,1+1:Nx2+1+1]-2*cell_cent_phi_dn[1:Nx1+1,1:Nx2+1]+cell_cent_phi_dn[1:Nx1+1,1-1:Nx2+1-1]


    Dtda_x_p_n[1:Nx1+1,1:Nx2+1] = Dd_x_p_n[1:Nx1+1,1:Nx2+1]-0.5*np.where((abs(DDd_pm_x_n[1:Nx1+1,1:Nx2+1])<abs(DDd_pm_x_n[1+1:Nx1+1+1,1:Nx2+1])),DDd_pm_x_n[1:Nx1+1,1:Nx2+1],DDd_pm_x_n[1+1:Nx1+1+1,1:Nx2+1])

    Dtda_x_m_n[1:Nx1+1,1:Nx2+1] = Dd_y_m_n[1:Nx1+1,1:Nx2+1]+0.5*np.where((abs(DDd_pm_x_n[1:Nx1+1,1:Nx2+1])<abs(DDd_pm_x_n[1-1:Nx1+1-1,1:Nx2+1])),DDd_pm_x_n[1:Nx1+1,1:Nx2+1],DDd_pm_x_n[1-1:Nx1+1-1,1:Nx2+1])

    Dtda_y_p_n[1:Nx1+1,1:Nx2+1]= Dd_y_p_n[1:Nx1+1,1:Nx2+1]-0.5*np.where((abs(DDd_pm_y_n[1:Nx1+1,1:Nx2+1])<abs(DDd_pm_y_n[1+1:Nx1+1+1,1:Nx2+1])),DDd_pm_y_n[1:Nx1+1,1:Nx2+1],DDd_pm_y_n[1+1:Nx1+1+1,1:Nx2+1])

    Dtda_y_m_n[1:Nx1+1,1:Nx2+1]= Dd_y_m_n[1:Nx1+1,1:Nx2+1]+0.5*np.where((abs(DDd_pm_y_n[1:Nx1+1,1:Nx2+1])<abs(DDd_pm_y_n[1-1:Nx1+1-1,1:Nx2+1])),DDd_pm_y_n[1:Nx1+1,1:Nx2+1],DDd_pm_y_n[1-1:Nx1+1-1,1:Nx2+1])


    Dtda_x_n[1:Nx1+1,1:Nx2+1] = np.where((sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_p_n[1:Nx1+1,1:Nx2+1]<0) & (sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_m_n[1:Nx1+1,1:Nx2+1]<-1*sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_p_n[1:Nx1+1,1:Nx2+1]), Dd_x_p_n[1:Nx1+1,1:Nx2+1],np.where((sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_p_n[1:Nx1+1,1:Nx2+1]<0) & (sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_p_n[1:Nx1+1,1:Nx2+1]>-1*sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_m_n[1:Nx1+1,1:Nx2+1]), Dd_x_m_n[1:Nx1+1,1:Nx2+1],np.where((sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_p_n[1:Nx1+1,1:Nx2+1]>0) & (sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_m_n[1:Nx1+1,1:Nx2+1]>0), 0.5*(Dtda_x_p_n[1:Nx1+1,1:Nx2+1]+Dtda_x_m_n[1:Nx1+1,1:Nx2+1]),0.5*(Dtda_x_p_n[1:Nx1+1,1:Nx2+1]+Dtda_x_m_n[1:Nx1+1,1:Nx2+1]))  ))  
    Dtda_y_n[1:Nx1+1,1:Nx2+1] = np.where((sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_p_n[1:Nx1+1,1:Nx2+1]<0) & (sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_m_n[1:Nx1+1,1:Nx2+1]<-1*sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_p_n[1:Nx1+1,1:Nx2+1]), Dd_y_p_n[1:Nx1+1,1:Nx2+1],np.where((sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_p_n[1:Nx1+1,1:Nx2+1]<0) & (sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_p_n[1:Nx1+1,1:Nx2+1]>-1*sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_m_n[1:Nx1+1,1:Nx2+1]), Dd_y_m_n[1:Nx1+1,1:Nx2+1],np.where((sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_p_n[1:Nx1+1,1:Nx2+1]>0) & (sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_m_n[1:Nx1+1,1:Nx2+1]>0), 0.5*(Dtda_y_p_n[1:Nx1+1,1:Nx2+1]+Dtda_y_m_n[1:Nx1+1,1:Nx2+1]),0.5*(Dtda_y_p_n[1:Nx1+1,1:Nx2+1]+Dtda_y_m_n[1:Nx1+1,1:Nx2+1]))  ))  
    
    return sign_phi_Mh[1:Nx1+1,1:Nx2+1]*(1-((Dtda_x_n[1:Nx1+1,1:Nx2+1]/h)**2+(Dtda_y_n[1:Nx1+1,1:Nx2+1]/h)**2)**0.5)
def L_phi_ds(cell_cent_phi_ds,cell_cent_phis,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    h=var_dict['h']
    M=var_dict['M']
    sign_phi=np.zeros_like(cell_cent_phis)
    sign_phi_Mh=np.zeros_like(cell_cent_phis)
    Dd_x_p_s=np.zeros_like(cell_cent_phi_ds)
    Dd_x_m_s=np.zeros_like(cell_cent_phi_ds)
    Dd_y_p_s=np.zeros_like(cell_cent_phi_ds)
    Dd_y_m_s=np.zeros_like(cell_cent_phi_ds)
    DDd_pm_x_s=np.zeros_like(cell_cent_phi_ds)
    DDd_pm_y_s=np.zeros_like(cell_cent_phi_ds)
    
    Dtda_x_p_s=np.zeros_like(Dd_x_p_s)
    Dtda_x_m_s=np.zeros_like(Dd_y_m_s)
    Dtda_y_p_s=np.zeros_like(Dd_y_p_s)
    Dtda_y_m_s=np.zeros_like(Dd_y_m_s)
    Dtda_x_s=np.zeros_like(Dd_x_m_s)
    Dtda_y_s=np.zeros_like(Dd_y_p_s)
    
    sign_phi[1:Nx1+1,1:Nx2+1]=np.where(cell_cent_phis[1:Nx1+1,1:Nx2+1]>0,1,np.where(cell_cent_phis[1:Nx1+1,1:Nx2+1]<0,-1,0))
    sign_phi_Mh[1:Nx1+1,1:Nx2+1]=np.where(cell_cent_phis[1:Nx1+1,1:Nx2+1]>(M*h),1,np.where(cell_cent_phis[1:Nx1+1,1:Nx2+1]<(-1*M*h),-1,cell_cent_phis[1:Nx1+1,1:Nx2+1]/(M*h)-np.sin((np.pi*cell_cent_phis[1:Nx1+1,1:Nx2+1])/(M*h))/np.pi))
    
    
    Dd_x_p_s[1:Nx1+1,1:Nx2+1]= cell_cent_phi_ds[1+1:Nx1+1+1,1:Nx2+1]-cell_cent_phi_ds[1:Nx1+1,1:Nx2+1]
    Dd_x_m_s[1:Nx1+1,1:Nx2+1]= cell_cent_phi_ds[1:Nx1+1,1:Nx2+1]-cell_cent_phi_ds[1-1:Nx1+1-1,1:Nx2+1]
    Dd_y_p_s[1:Nx1+1,1:Nx2+1]= cell_cent_phi_ds[1:Nx1+1,1+1:Nx2+1+1]-cell_cent_phi_ds[1:Nx1+1,1:Nx2+1]
    Dd_y_m_s[1:Nx1+1,1:Nx2+1]= cell_cent_phi_ds[1:Nx1+1,1:Nx2+1]-cell_cent_phi_ds[1:Nx1+1,1-1:Nx2+1-1]
    DDd_pm_x_s[Nx1+1,1:Nx2+1] =  cell_cent_phi_ds[2,1:Nx2+1]-2*cell_cent_phi_ds[Nx1+1,1:Nx2+1]+cell_cent_phi_ds[Nx1+1-1,1:Nx2+1]    
    DDd_pm_x_s[0,1:Nx2+1]= cell_cent_phi_ds[0+1,1:Nx2+1]-2*cell_cent_phi_ds[0,1:Nx2+1]+cell_cent_phi_ds[-3,1:Nx2+1]
    DDd_pm_x_s[1:Nx1+1,1:Nx2+1]= cell_cent_phi_ds[1+1:Nx1+1+1,1:Nx2+1]-2*cell_cent_phi_ds[1:Nx1+1,1:Nx2+1]+cell_cent_phi_ds[1-1:Nx1+1-1,1:Nx2+1]
    DDd_pm_y_s[Nx1+1,1:Nx2+1] =  cell_cent_phi_ds[2,1:Nx2+1]-2*cell_cent_phi_ds[Nx1+1,1:Nx2+1]+cell_cent_phi_ds[Nx1+1-1,1:Nx2+1]
    DDd_pm_y_s[1:Nx1+1,0]= cell_cent_phi_ds[1:Nx1+1,-1]-2*cell_cent_phi_ds[1:Nx1+1,-2]+cell_cent_phi_ds[1:Nx1+1,-3]
    DDd_pm_y_s[1:Nx1+1,1:Nx2+1]= cell_cent_phi_ds[1:Nx1+1,1+1:Nx2+1+1]-2*cell_cent_phi_ds[1:Nx1+1,1:Nx2+1]+cell_cent_phi_ds[1:Nx1+1,1-1:Nx2+1-1]
    Dtda_x_p_s[1:Nx1+1,1:Nx2+1] = Dd_x_p_s[1:Nx1+1,1:Nx2+1]-0.5*np.where((abs(DDd_pm_x_s[1:Nx1+1,1:Nx2+1])<abs(DDd_pm_x_s[1+1:Nx1+1+1,1:Nx2+1])),DDd_pm_x_s[1:Nx1+1,1:Nx2+1],DDd_pm_x_s[1+1:Nx1+1+1,1:Nx2+1])

    Dtda_x_m_s[1:Nx1+1,1:Nx2+1] = Dd_y_m_s[1:Nx1+1,1:Nx2+1]+0.5*np.where((abs(DDd_pm_x_s[1:Nx1+1,1:Nx2+1])<abs(DDd_pm_x_s[1-1:Nx1+1-1,1:Nx2+1])),DDd_pm_x_s[1:Nx1+1,1:Nx2+1],DDd_pm_x_s[1-1:Nx1+1-1,1:Nx2+1])

    Dtda_y_p_s[1:Nx1+1,1:Nx2+1]= Dd_y_p_s[1:Nx1+1,1:Nx2+1]-0.5*np.where((abs(DDd_pm_y_s[1:Nx1+1,1:Nx2+1])<abs(DDd_pm_y_s[1+1:Nx1+1+1,1:Nx2+1])),DDd_pm_y_s[1:Nx1+1,1:Nx2+1],DDd_pm_y_s[1+1:Nx1+1+1,1:Nx2+1])

    Dtda_y_m_s[1:Nx1+1,1:Nx2+1]= Dd_y_m_s[1:Nx1+1,1:Nx2+1]+0.5*np.where((abs(DDd_pm_y_s[1:Nx1+1,1:Nx2+1])<abs(DDd_pm_y_s[1-1:Nx1+1-1,1:Nx2+1])),DDd_pm_y_s[1:Nx1+1,1:Nx2+1],DDd_pm_y_s[1-1:Nx1+1-1,1:Nx2+1])

    Dtda_x_s[1:Nx1+1,1:Nx2+1] = np.where((sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_p_s[1:Nx1+1,1:Nx2+1]<0) & (sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_m_s[1:Nx1+1,1:Nx2+1]<-1*sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_p_s[1:Nx1+1,1:Nx2+1]), Dd_x_p_s[1:Nx1+1,1:Nx2+1],np.where((sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_p_s[1:Nx1+1,1:Nx2+1]<0) & (sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_p_s[1:Nx1+1,1:Nx2+1]>-1*sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_m_s[1:Nx1+1,1:Nx2+1]), Dd_x_m_s[1:Nx1+1,1:Nx2+1],np.where((sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_p_s[1:Nx1+1,1:Nx2+1]>0) & (sign_phi[1:Nx1+1,1:Nx2+1]*Dd_x_m_s[1:Nx1+1,1:Nx2+1]>0), 0.5*(Dtda_x_p_s[1:Nx1+1,1:Nx2+1]+Dtda_x_m_s[1:Nx1+1,1:Nx2+1]),0.5*(Dtda_x_p_s[1:Nx1+1,1:Nx2+1]+Dtda_x_m_s[1:Nx1+1,1:Nx2+1]))  ))  
    Dtda_y_s[1:Nx1+1,1:Nx2+1] = np.where((sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_p_s[1:Nx1+1,1:Nx2+1]<0) & (sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_m_s[1:Nx1+1,1:Nx2+1]<-1*sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_p_s[1:Nx1+1,1:Nx2+1]), Dd_y_p_s[1:Nx1+1,1:Nx2+1],np.where((sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_p_s[1:Nx1+1,1:Nx2+1]<0) & (sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_p_s[1:Nx1+1,1:Nx2+1]>-1*sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_m_s[1:Nx1+1,1:Nx2+1]), Dd_y_m_s[1:Nx1+1,1:Nx2+1],np.where((sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_p_s[1:Nx1+1,1:Nx2+1]>0) & (sign_phi[1:Nx1+1,1:Nx2+1]*Dd_y_m_s[1:Nx1+1,1:Nx2+1]>0), 0.5*(Dtda_y_p_s[1:Nx1+1,1:Nx2+1]+Dtda_y_m_s[1:Nx1+1,1:Nx2+1]),0.5*(Dtda_y_p_s[1:Nx1+1,1:Nx2+1]+Dtda_y_m_s[1:Nx1+1,1:Nx2+1]))  ))  
    
    return sign_phi_Mh[1:Nx1+1,1:Nx2+1]*(1-((Dtda_x_s[1:Nx1+1,1:Nx2+1]/h)**2+(Dtda_y_s[1:Nx1+1,1:Nx2+1]/h)**2)**0.5)
def f_st_x(cell_cent_phin,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    M=var_dict['M']
    h=var_dict['h']
    dfdphi=np.zeros_like(cell_cent_phin)
    grad_phi_x=np.zeros_like(cell_cent_phin)
    grad_phi_xhpp=np.zeros_like(cell_cent_phin)
    grad_phi_xhpm=np.zeros_like(cell_cent_phin)
    grad_phi_xhmp=np.zeros_like(cell_cent_phin)
    grad_phi_xhmm=np.zeros_like(cell_cent_phin)
    grad_phi_yhpp=np.zeros_like(cell_cent_phin)
    grad_phi_yhpm=np.zeros_like(cell_cent_phin)
    grad_phi_yhmp= np.zeros_like(cell_cent_phin)
    grad_phi_yhmm= np.zeros_like(cell_cent_phin)
    grad_phi_mag_hpp=np.zeros_like(cell_cent_phin)
    grad_phi_mag_hpm=np.zeros_like(cell_cent_phin)
    grad_phi_mag_hmp=np.zeros_like(cell_cent_phin)
    grad_phi_mag_hmm=np.zeros_like(cell_cent_phin)
    kappa=np.zeros_like(cell_cent_phin)
    
    grad_phi_xhpp[1:Nx1+1,1:Nx2+1]= (cell_cent_phin[1+1:Nx1+1+1,1+1:Nx2+1+1]+cell_cent_phin[1+1:Nx1+1+1,1:Nx2+1]-cell_cent_phin[1:Nx1+1,1+1:Nx2+1+1]-cell_cent_phin[1:Nx1+1,1:Nx2+1])/(2*h)
    grad_phi_xhpm[1:Nx1+1,1:Nx2+1]= (cell_cent_phin[1+1:Nx1+1+1,1:Nx2+1]+cell_cent_phin[1+1:Nx1+1+1,1-1:Nx2+1-1]-cell_cent_phin[1:Nx1+1,1:Nx2+1]-cell_cent_phin[1:Nx1+1,1-1:Nx2+1-1])/(2*h)
    grad_phi_xhmp[1:Nx1+1,1:Nx2+1]= (cell_cent_phin[1:Nx1+1,1+1:Nx2+1+1]+cell_cent_phin[1:Nx1+1,1:Nx2+1]-cell_cent_phin[1-1:Nx1+1-1,1+1:Nx2+1+1]-cell_cent_phin[1-1:Nx1+1-1,1:Nx2+1])/(2*h)
    grad_phi_xhmm[1:Nx1+1,1:Nx2+1]= (cell_cent_phin[1:Nx1+1,1:Nx2+1]+cell_cent_phin[1:Nx1+1,1-1:Nx2+1-1]-cell_cent_phin[1-1:Nx1+1-1,1:Nx2+1]-cell_cent_phin[1-1:Nx1+1-1,1-1:Nx2+1-1])/(2*h)
    grad_phi_yhpp[1:Nx1+1,1:Nx2+1]= (cell_cent_phin[1+1:Nx1+1+1,1+1:Nx2+1+1]-cell_cent_phin[1+1:Nx1+1+1,1:Nx2+1]+cell_cent_phin[1:Nx1+1,1+1:Nx2+1+1]-cell_cent_phin[1:Nx1+1,1:Nx2+1])/(2*h)
    grad_phi_yhpm[1:Nx1+1,1:Nx2+1]= (cell_cent_phin[1+1:Nx1+1+1,1:Nx2+1]-cell_cent_phin[1+1:Nx1+1+1,1-1:Nx2+1-1]+cell_cent_phin[1:Nx1+1,1:Nx2+1]-cell_cent_phin[1:Nx1+1,1-1:Nx2+1-1])/(2*h)
    grad_phi_yhmp[1:Nx1+1,1:Nx2+1]= (cell_cent_phin[1:Nx1+1,1+1:Nx2+1+1]-cell_cent_phin[1:Nx1+1,1:Nx2+1]+cell_cent_phin[1-1:Nx1+1-1,1+1:Nx2+1+1]-cell_cent_phin[1-1:Nx1+1-1,1:Nx2+1])/(2*h)

    grad_phi_yhmm[1:Nx1+1,1:Nx2+1]= (cell_cent_phin[1:Nx1+1,1:Nx2+1]-cell_cent_phin[1:Nx1+1,1-1:Nx2+1-1]+cell_cent_phin[1-1:Nx1+1-1,1:Nx2+1]+cell_cent_phin[1-1:Nx1+1-1,1-1:Nx2+1-1])/(2*h)

    grad_phi_mag_hpp[1:Nx1+1,1:Nx2+1]= (grad_phi_xhpp[1:Nx1+1,1:Nx2+1])**2+(grad_phi_yhpp[1:Nx1+1,1:Nx2+1]**2)**0.5

    grad_phi_mag_hpm[1:Nx1+1,1:Nx2+1]= (grad_phi_xhpm[1:Nx1+1,1:Nx2+1])**2+(grad_phi_yhpm[1:Nx1+1,1:Nx2+1]**2)**0.5

    grad_phi_mag_hmp[1:Nx1+1,1:Nx2+1]= (grad_phi_xhmp[1:Nx1+1,1:Nx2+1])**2+(grad_phi_yhmp[1:Nx1+1,1:Nx2+1]**2)**0.5

    grad_phi_mag_hmm[1:Nx1+1,1:Nx2+1]= (grad_phi_xhmm[1:Nx1+1,1:Nx2+1])**2+(grad_phi_yhmm[1:Nx1+1,1:Nx2+1]**2)**0.5

    kappa[1:Nx1+1,1:Nx2+1]= ((grad_phi_xhpp[1:Nx1+1,1:Nx2+1]/grad_phi_mag_hpp[1:Nx1+1,1:Nx2+1])+(grad_phi_xhpm[1:Nx1+1,1:Nx2+1]/grad_phi_mag_hpm[1:Nx1+1,1:Nx2+1])-(grad_phi_xhmp[1:Nx1+1,1:Nx2+1]/grad_phi_mag_hmp[1:Nx1+1,1:Nx2+1])-(grad_phi_xhmm[1:Nx1+1,1:Nx2+1]/grad_phi_mag_hmm[1:Nx1+1,1:Nx2+1])+(grad_phi_yhpp[1:Nx1+1,1:Nx2+1]/grad_phi_mag_hpp[1:Nx1+1,1:Nx2+1])-(grad_phi_yhpm[1:Nx1+1,1:Nx2+1]/grad_phi_mag_hpm[1:Nx1+1,1:Nx2+1])+(grad_phi_yhmp[1:Nx1+1,1:Nx2+1]/grad_phi_mag_hmp[1:Nx1+1,1:Nx2+1])-(grad_phi_yhmm[1:Nx1+1,1:Nx2+1]/grad_phi_mag_hmm[1:Nx1+1,1:Nx2+1]))

    
    dfdphi[1:Nx1+1,1:Nx2+1]= np.where(cell_cent_phin[1:Nx1+1,1:Nx2+1]<-1*M*h or cell_cent_phin[1:Nx1+1,1:Nx2+1]>1*M*h,0,(1+np.cos((np.pi*cell_cent_phin[1:Nx1+1,1:Nx2+1])/(M*h)))/(2*M*h))
    grad_phi_x[1:Nx1+1,1:Nx2+1]= (cell_cent_phin[1+1:Nx1+1+1,1: Nx2+1]-cell_cent_phin[1-1:Nx1+1-1,1: Nx2+1])/(2*var_dict['h'])
    return var_dict['st_coef']*kappa[1:Nx1+1,1:Nx2+1]*dfdphi[1:Nx1+1,1:Nx2+1]*grad_phi_x[1:Nx1+1,1:Nx2+1]


def f(cell_cent_phin,var_dict):
    h=var_dict['h']
    M=var_dict['M']
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    return np.where(cell_cent_phin[0:Nx1+2,0:Nx2+2]<-1*M*h,0,np.where(cell_cent_phin[0:Nx1+2,0:Nx2+2]>M*h,1,0.5*(1+(cell_cent_phin[0:Nx1+2,0:Nx2+2])/(M*h)+(np.sin((np.pi*cell_cent_phin)/(M*h)))/np.pi)))
def rho_distr(cell_cent_phin,var_dict):
    rho_in=1000
    rho_out=999
    return rho_in*f(cell_cent_phin,var_dict)+rho_out*(1-f(cell_cent_phin,var_dict))
def mu_distr(cell_cent_phin,var_dict):
    mu_in=1e-3
    mu_out=1e-4
    return mu_in*f(cell_cent_phin,var_dict)+mu_out*(1-f(cell_cent_phin,var_dict))