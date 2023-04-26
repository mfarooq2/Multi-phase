import itertools
import numpy as np
from util_tools.helper import *
import matplotlib.pyplot as plt
from util_tools.operators import *
from util_tools.update_funcs import *
#problem constants
var_dict = {}
nu_c=var_dict['nu_c']=1e-6
mu_c=var_dict['mu_c']=1e-3
rho_c=var_dict['rho_c']=1e+3
st_coef=0.06
#real timestep
dt=0.000001
gradP=-2.4
from tqdm import tqdm
n_iter=0
global epstot
'''
node generation section
'''
#domain length
nx2=50
Lx1=0.04;Lx2=0.02;r_dpl=Lx2/4;M=3.0;u1_ave=0.0125;nx1=nx2*2;Nx1=nx1+1;Nx2=nx2+2
h=Lx1/nx1
var_dict = {'Lx1' : Lx1, 'Lx2' : Lx2, 'r_dpl': r_dpl,'h': h,'u1_ave':u1_ave,'Nx1':Nx1,'Nx2':Nx2,'M' : M,'nx1':nx1,'nx2':nx2,'dt':dt,'gradP':gradP,'st_coef':st_coef,'nu_c':nu_c,'mu_c':mu_c,'rho_c':rho_c}

#Initialization
un=np.zeros([Nx1,Nx2])
us=np.zeros([Nx1,Nx2])
unn=np.zeros([Nx1,Nx2])
u_ref=np.zeros([Nx1,Nx2])

vn=np.zeros([Nx1,Nx2])
vs=np.zeros([Nx1,Nx2])
vnn=np.zeros([Nx1,Nx2])

pn=np.zeros([nx1,nx2])
pnn=np.zeros([nx1,nx2])


rho=np.zeros([nx1,nx2])
mu=np.zeros([nx1,nx2])



phin=np.zeros( [nx1,nx2] )
phis=np.zeros([nx1,nx2])
phinn=np.zeros([nx1,nx2])

Tx1,Tx2=17,15

#half_index_grid
hig_x, hig_y = np.meshgrid(np.linspace(-Lx1/2, Lx1/2, num=Nx1),np.linspace(-Lx2/2, Lx2/2,Nx2),indexing='ij')
xi=np.round(inter_polator(hig_x,var_dict),4)
xj=np.round(inter_polator(hig_y,var_dict),4)

# #full_index_grid
#xi, xj = np.meshgrid(np.round(np.linspace(0+h/2, Lx1-h/2, num=nx1),4),np.round(np.linspace(0+h/2, Lx2-h/2,nx2),4),indexing='ij')

# #lvlset init
phin_int=lvlset_init(xi, xj,var_dict)
phin=phin_int.copy()

rho_int=rho_distr(phin,var_dict)
rho=rho_int.copy()

mu_int=mu_distr(phin,var_dict)
mu=mu_int.copy()

u=ref_vel_prof(hig_y[0])
for _ in range(Nx1):
    un[_,:]=u
#un[:]
u_inter=inter_polator(un,var_dict)


for _ in tqdm(range(50)):
    us=predictor(un,vn, mu, rho,var_dict)
    us=BC_looper(us)
    res=1000
    while res>0.5e-3:
        pnn=projector( us, vs, pn,rho,var_dict)
        res1=res
        res=LA.norm(pnn-pn)
        pn=pnn.copy()
    unn=corrector(us, pnn, unn, rho,var_dict)
    unn=BC_looper(unn)

    L_phi_n,phis=phis_predictor(phin,un,vn,var_dict)

    phinn=phinn_corrector(L_phi_n,phin,phis,un,vn,var_dict)

    
    rho=rho_distr(phin,var_dict)
    mu=mu_distr(phin,var_dict)
    tr=np.where(unn==0,0.0001,unn)
    tr=np.where((tr<=1e5)&(tr>=-1e5),tr,False)
    if np.all(tr)==False:
        break
    tr=np.where(pnn==0,0.0001,pnn)
    tr=np.where((tr<=1e5)&(tr>=-1e5),tr,False)
    if np.all(tr)==False:
        break
    un=unn.copy()
    phin=phinn.copy()
    #var_dict['dt']=var_dict['dt']/5 
u=inter_polator(un,var_dict)

drop_elem = np.where(phin <0,np.min(phin),0)
drop_elem_int = np.where(phin_int<0,np.min(phin_int),0)

#phin[np.where(phin > 0)]