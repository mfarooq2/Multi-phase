import numpy as np

def inter_polator(qi,var_dict):
    q_out=np.zeros((var_dict['nx1'],var_dict['nx2']))
    q_out[:,:]=(qi[1:,1:-1]+qi[0:-1,1:-1])/2  
    return q_out