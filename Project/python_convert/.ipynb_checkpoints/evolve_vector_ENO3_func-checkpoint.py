import numpy as np
from upwind_ENO3_func import upwind_ENO3
def evolve_vector_ENO3(phi, dx, dy, u_ext, v_ext):
    """
    Finds the amount of evolution under a vector field
    based force and using 3rd order accurate ENO scheme
    """
    delta = np.zeros((np.shape(phi)[0]+6,np.shape(phi)[1]+6));
    data_ext = np.zeros((np.shape(phi)[0]+6,np.shape(phi)[1]+6));
    data_ext[3:-3,3:-3] = phi;
    # first scan the rows
    for i in range(phi.shape[0]):
        delta[i+3,:] = delta[i+3,:] + upwind_ENO3(data_ext[i+3,:], u_ext, dx);
    # then scan the columns
    for j in range(phi.shape[1]):
        delta[:,j+3] = delta[:,j+3] + upwind_ENO3(data_ext[:,j+3], v_ext, dy);
    delta = delta[3:-3,3:-3];
    return delta