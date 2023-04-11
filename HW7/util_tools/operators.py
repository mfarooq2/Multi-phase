

import math
import numpy as np
def Adv_x_n(un,vn,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    h=var_dict['h']
    advxn=np.zeros([Nx1+2,Nx2+2])
    advxn[1:Nx1+1,1:Nx2+1]=(1/h)*((0.5*(un[1:Nx1+1,1:Nx2+1]+un[1+1:Nx1+1+1,1:Nx2+1]))**2 - (0.5*(un[1:Nx1+1,1:Nx2+1]+un[1-1:Nx1+1-1,1:Nx2+1]))**2)
                                  ##+(0.5*(un[1:Nx1+1,1+1:Nx2+1+1]+un[1:Nx1+1,1:Nx2+1]))*(vn[1:Nx1+1,1+1:Nx2+1+1]+vn[1+1:Nx1+1+1,1+1:Nx2+1+1])-(0.5*(vn[1:Nx1+1,1:Nx2+1]+vn[1:Nx1+1,1-1:Nx2+1-1]))*(0.5*(vn[1:Nx1+1,1:Nx2+1]+vn[1+1:Nx1+1+1,1:Nx2+1]))) 
    return advxn

def Dif_x_n(un,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    h=var_dict['h']
    difxn=np.zeros([Nx1+2,Nx2+2])
    difxn[1:Nx1+1,1:Nx2+1]=(1/(h**2))*(un[1+1:Nx1+1+1,1:Nx2+1]+un[1-1:Nx1+1-1,1:Nx2+1]+un[1:Nx1+1,1+1:Nx2+1+1]+un[1:Nx1+1,1-1:Nx2+1-1]-4*un[1:Nx1+1,1:Nx2+1])
    return difxn
    
    
#def Adv_y_n(i,j):
#    return (1/h)*((0.5*())**2-(0.5*())**2+(0.5*())*(0.5*())-(0.5*())*(0.5*()))

def Dif_y_n(vn,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    h=var_dict['h']
    return (
        1
        / (h**2)
        * (
            vn[1 + 1 : Nx1 + 1 + 1, 1 : Nx2 + 1]
            + vn[0 : Nx1+ 1 - 1, 1 : Nx2 + 1]
            + vn[1 : Nx1 + 1, 1 + 1 : Nx2 + 1 + 1]
            + vn[1 : Nx1 + 1, 0 : Nx2 + 1 - 1]
            - 4 * vn[1 : Nx1 + 1, 1 : Nx2 + 1]
        )
    )

def ref_vel_prof(x2):
    return -1200*((x2-0.005)**2)+0.03
    

def lvlset_init(x,y,var_dict):
    return -1*(np.sqrt((x-0.01)**2+(y-0.005)**2)-var_dict['r_dpl'])

#level-set functions
def L_phi(phi,u,v,var_dict):
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    h=var_dict['h']
    D_x_p=np.zeros_like(phi)
    D_x_m=np.zeros_like(phi)
    phi_xh=np.zeros_like(u)
    phi_hx=np.zeros_like(u)
    D_x_p[Nx1+1,1:Nx2+1] = phi[2,1:Nx2+1]-phi[Nx1+1,1:Nx2+1]
    D_x_p[1:Nx1+1,1:Nx2+1]= phi[1+1:Nx1+1+1,1:Nx2+1]-phi[1:Nx1+1,1:Nx2+1]
    
    D_x_m[Nx1+1,1:Nx2+1]= phi[2,1:Nx2+1]-phi[Nx1+1-1,1:Nx2+1]
    D_x_m[1:Nx1+1,1:Nx2+1]= phi[1:Nx1+1,1:Nx2+1]-phi[1-1:Nx1+1-1,1:Nx2+1]
    #For i+1/2,j Eq 3.44
    for j in range(1, int(Nx2+1)):
        for i in range(1, int(Nx1+1)):
            if 0.5*(u[i+1,j]+u[i,j])>0:
                phi_xh[i,j]= phi[i,j]+0.5*np.where(abs(D_x_p[i,j])< abs(D_x_m[i,j]),D_x_p[i,j], D_x_m[i,j])      
            elif 0.5*(u[i+1,j]+u[i,j])<0:
                phi_xh[i+1,j]= u[i+1,j]-0.5*np.where(abs(D_x_p[i+1,j])< abs(D_x_m[i+1,j]),D_x_p[i+1,j], D_x_m[i+1,j])      
                
    for j in range(1, int(Nx2+1)):
        for i in range(1, int(Nx1+1)):                                                          #For i-1/2,j Eq 3.44
            if 0.5*(u[i,j]+u[i-1,j])<0:
                phi_hx[i,j]= phi[i,j]-0.5*np.where(abs(D_x_p[i,j])< abs(D_x_m[i,j]),D_x_p[i,j], D_x_m[i,j]) 
            elif 0.5*(u[i,j]+u[i-1,j])>0:
                phi_hx[i-1,j]= phi[i-1,j]+0.5*np.where(abs(D_x_p[i-1,j])< abs(D_x_m[i-1,j]),D_x_p[i-1,j], D_x_m[i-1,j]) 
    
    #phi_xh[1:Nx1+1,1:Nx2+1]=np.where(0.5*(u[1+1:Nx1+1+1,1:Nx2+1]+u[1:Nx1+1,1:Nx2+1])>0,phi[1:Nx1+1,1:Nx2+1] + 0.5*np.where((abs(D_x_p[1:Nx1+1,1:Nx2+1])<abs(D_x_m[1:Nx1+1,1:Nx2+1])),D_x_p[1:Nx1+1,1:Nx2+1],D_x_m[1:Nx1+1,1:Nx2+1]),phi[1+1:Nx1+1+1,1:Nx2+1] - 0.5*np.where((abs(D_x_p[1+1:Nx1+1+1,1:Nx2+1])<abs(D_x_m[1+1:Nx1+1+1,1:Nx2+1])),D_x_p[1+1:Nx1+1+1,1:Nx2+1],D_x_m[1+1:Nx1+1+1,1:Nx2+1]))
    #phi_hx[1:Nx1+1,1:Nx2+1]=np.where(0.5*(u[1:Nx1+1,1:Nx2+1]+u[1-1:Nx1+1-1,1:Nx2+1])>0,phi[1:Nx1+1,1:Nx2+1] + 0.5*np.where((abs(D_x_p[1-1:Nx1+1-1,1:Nx2+1])<abs(D_x_m[1-1:Nx1+1-1,1:Nx2+1])),D_x_p[1-1:Nx1+1-1,1:Nx2+1],D_x_m[1-1:Nx1+1-1,1:Nx2+1]),phi[1:Nx1+1,1:Nx2+1] - 0.5*np.where((abs(D_x_p[1:Nx1+1,1:Nx2+1])<abs(D_x_m[1:Nx1+1,1:Nx2+1])),D_x_p[1:Nx1+1,1:Nx2+1],D_x_m[1:Nx1+1,1:Nx2+1]))
    return -1*(u)*(phi_xh-phi_hx)/h

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


def f(phin,var_dict):
    h=var_dict['h']
    M=var_dict['M']
    Nx1=var_dict['Nx1']
    Nx2=var_dict['Nx2']
    return np.where(phin<-1*M*h,0,np.where(phin>M*h,1,0.5*(1+(phin)/(M*h)+(np.sin((np.pi*phin)/(M*h)))/np.pi)))
def rho_distr(phin,var_dict):
    rho_in=1000
    rho_out=1
    return rho_in*f(phin,var_dict)+rho_out*(1-f(phin,var_dict))
def mu_distr(phin,var_dict):
    mu_in=1e-3
    mu_out=1e-5
    return mu_in*f(phin,var_dict)+mu_out*(1-f(phin,var_dict))