import numpy as np
import matplotlib.pyplot as plt
from curvature_func import curvature
from normcd_func import normcd
from laplacian_func import laplacian
from evolve_vector_ENO3_func import evolve_vector_ENO3
from upwind_ENO3_func import upwind_ENO3
from tqdm import tqdm
def evolve_vector(phi, h, u_ext, v_ext):
    bp = 0.5
    dt=0.001
    tf=0.5;
    W = 2 * h
    b = W * bp
    W2_inv = 1 / (W ** 2)
    for _ in tqdm(np.arange(0,tf,dt)):
        kappa= curvature(phi, h)
        normphi = normcd(phi,h)
        d2phi = laplacian(phi, h)
        delta = evolve_vector_ENO3(phi, h, h, u_ext, v_ext)
        mid_express=d2phi[1:-1, 1:-1] + W2_inv * phi[1:-1, 1:-1] * (1 - phi[1:-1, 1:-1] ** 2) - normphi[1:-1, 1:-1] * kappa[1:-1, 1:-1]
        phi[1:-1, 1:-1] = phi[1:-1, 1:-1] + dt * (b * (mid_express) - delta[1:-1, 1:-1])
    return phi
        
def devolve_vector(phi, h, u_ext, v_ext):
    bp = 0.5
    dt=0.001
    tf=0.5;
    W = 2 * h
    b = W * bp
    W2_inv = 1 / (W ** 2)
    for _ in tqdm(np.arange(0,tf,dt)):
        kappa= curvature(phi, h)
        normphi = normcd(phi,h);
        d2phi = laplacian(phi, h)
        delta = evolve_vector_ENO3(phi, h, h, u_ext, v_ext)
        phi[1:-1, 1:-1] = phi[1:-1, 1:-1] + dt * (b * (d2phi[1:-1, 1:-1] + W2_inv * phi[1:-1, 1:-1] * (1 - phi[1:-1, 1:-1] ** 2) - normphi[1:-1, 1:-1] * kappa[1:-1, 1:-1]) - delta[1:-1, 1:-1])
    return phi