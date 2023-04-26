# % A simple distance function evaluator from a point with coordinates(x,y)
# % to a given set of points S.
import numpy as np
def distance_function (x,y,S):
    dmin=np.inf
    for i in range(0,len(S)):
        dx = abs(S[i,0]-x)
        dy = abs(S[i,1]-y)
        d = np.sqrt(dx**2+dy**2)
        if d<dmin:
            dmin=d
    return dmin