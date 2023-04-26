import numpy as np
def C_d_Func(rho_l,D_b,u_l,gamma,v_r,rho_g,g):
    Re_b=(rho_l*D_b*v_r)/(u_l)
    E_o=(rho_l-rho_g)*g*((D_b)**2)/gamma
    return np.sqrt(((16/Re_b)*(1+(2)/(1+(16/Re_b)+(3.315/Re_b**0.5))))**2+((4*E_o)/(E_o+9.5))**2)