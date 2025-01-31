import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class SimulationParams:
    nx: int = 60
    ny: int = 120
    Lx: float = 0.6
    Ly: float = 1.2

    # Smaller dt for more stable, slower-moving simulation
    dt: float = 1e-4
    # Shorter total_time to avoid a huge number of steps
    total_time: float = 0.01

    # Reinitialize only once per step
    reinit_steps: int = 1
    reinit_dt: float = None  # set in __post_init__ if None

    # Fluid properties
    rho_inner: float = 1.0
    rho_outer: float = 1000.0
    mu_inner: float  = 0.1
    mu_outer: float  = 1.0

    # Reduced surface tension
    sigma: float     = 0.01
    
    g: float         = -9.81

    # Interface thickness factor
    epsilon_factor: float = 1.5

    # Initial bubble
    radius: float = 0.1
    x0: float = None
    y0: float = None

    def __post_init__(self):
        self.dx = self.Lx / self.nx
        self.dy = self.Ly / self.ny
        self.nsteps = int(self.total_time / self.dt)
        self.epsilon = self.epsilon_factor * min(self.dx, self.dy)

        # Make reinit_dt smaller to reduce reinit blow-up
        if self.reinit_dt is None:
            self.reinit_dt = 0.005 * min(self.dx, self.dy)

        if self.x0 is None:
            self.x0 = 0.5 * self.Lx
        if self.y0 is None:
            self.y0 = 0.4 * self.Ly

class Grid:
    """Manages grid coordinates and fields."""
    def __init__(self, params: SimulationParams):
        self.params = params
        # 1D arrays for cell centers
        self.xc = np.linspace(params.dx/2, params.Lx - params.dx/2, params.nx)
        self.yc = np.linspace(params.dy/2, params.Ly - params.dy/2, params.ny)
        # 2D mesh
        self.Xc, self.Yc = np.meshgrid(self.xc, self.yc, indexing='ij')

        # Staggered velocities
        self.u = np.zeros((params.nx+1, params.ny))
        self.v = np.zeros((params.nx, params.ny+1))

        # Pressure at cell centers
        self.pressure = np.zeros((params.nx, params.ny))

        # Level set
        self.phi = self.initialize_levelset()

    def initialize_levelset(self):
        return (np.sqrt((self.Xc - self.params.x0)**2 +
                        (self.Yc - self.params.y0)**2)
                - self.params.radius)

class LevelSetSolver:
    """Handles level set advection and reinitialization."""
    @staticmethod
    def reinitialize(phi: np.ndarray, phi0: np.ndarray, params: SimulationParams) -> np.ndarray:
        """
        PDE-based reinit with a Godunov scheme:
           dphi/dtau = sign(phi0)*(1 - |grad(phi)|).
        Includes a clamp on phi to avoid unbounded growth.
        """
        dx, dy = params.dx, params.dy
        dta = params.reinit_dt

        def sign_phi(val, eps=1e-12):
            return val / np.sqrt(val*val + eps)

        def upwind_x(ph, i, j):
            im1 = max(i-1, 0)
            ip1 = min(i+1, ph.shape[0]-1)
            return (ph[i,j] - ph[im1,j]) / dx, (ph[ip1,j] - ph[i,j]) / dx

        def upwind_y(ph, i, j):
            jm1 = max(j-1, 0)
            jp1 = min(j+1, ph.shape[1]-1)
            return (ph[i,j] - ph[i,jm1]) / dy, (ph[i,jp1] - ph[i,j]) / dy

        phi_new = phi.copy()
        band_width = 3.0 * params.epsilon  # clamp distance

        for _ in range(params.reinit_steps):
            phi_new[...] = phi
            # Copy boundaries
            phi_new[0,:]   = phi[0,:]
            phi_new[-1,:]  = phi[-1,:]
            phi_new[:,0]   = phi[:,0]
            phi_new[:,-1]  = phi[:,-1]

            for i in range(1, params.nx-1):
                for j in range(1, params.ny-1):
                    # Only reinit if |phi0| < band_width
                    if abs(phi0[i,j]) > band_width:
                        # Keep it or clamp to sign(phi0)*band_width
                        s = np.sign(phi0[i,j])
                        phi_new[i,j] = s*band_width
                        continue

                    s0 = sign_phi(phi0[i,j])
                    dxm, dxp = upwind_x(phi, i, j)
                    dym, dyp = upwind_y(phi, i, j)

                    # Godunov
                    if s0 > 0:
                        gx = max(max(dxm,0)**2, min(dxp,0)**2)
                        gy = max(max(dym,0)**2, min(dyp,0)**2)
                    else:
                        gx = max(min(dxm,0)**2, max(dxp,0)**2)
                        gy = max(min(dym,0)**2, max(dyp,0)**2)

                    grad_phi = np.sqrt(gx + gy + 1e-12)
                    phi_new[i,j] = phi[i,j] - dta*s0*(grad_phi - 1.0)

            phi[:] = phi_new

        # Final clamp
        phi = np.clip(phi, -band_width, band_width)
        return phi

    @staticmethod
    def advect(phi: np.ndarray, u: np.ndarray, v: np.ndarray, params: SimulationParams) -> np.ndarray:
        """
        First-order upwind advection of phi:
            phi^{n+1} = phi^n - dt*(u dphi/dx + v dphi/dy).
        Clamps phi at the end to avoid blow-up.
        """
        nx, ny = params.nx, params.ny
        dx, dy, dt = params.dx, params.dy, params.dt
        phi_new = phi.copy()

        # Velocity at cell centers
        uc = 0.5*(u[:-1,:] + u[1:,:])   # shape (nx, ny)
        vc = 0.5*(v[:,:-1] + v[:,1:])   # shape (nx, ny)

        band_width = 3.0 * params.epsilon

        for i in range(1, nx-1):
            for j in range(1, ny-1):
                velx = uc[i,j]
                vely = vc[i,j]

                # x upwind
                if velx > 0:
                    dphi_dx = (phi[i,j] - phi[i-1,j]) / dx
                else:
                    dphi_dx = (phi[i+1,j] - phi[i,j]) / dx

                # y upwind
                if vely > 0:
                    dphi_dy = (phi[i,j] - phi[i,j-1]) / dy
                else:
                    dphi_dy = (phi[i,j+1] - phi[i,j]) / dy

                phi_new[i,j] = phi[i,j] - dt*(velx*dphi_dx + vely*dphi_dy)

        # Clamp
        phi_new = np.clip(phi_new, -band_width, band_width)
        return phi_new

class FluidSolver:
    def __init__(self, params: SimulationParams):
        self.p = params

    def compute_curvature_and_delta(self, phi):
        dx, dy = self.p.dx, self.p.dy
        eps = self.p.epsilon

        # 1) Clip phi for stable gradient calculations
        band_width = 3.0 * eps
        phi_clipped = np.clip(phi, -band_width, band_width)

        # 2) Grad of phi
        phi_x = (np.roll(phi_clipped, -1, axis=0) - np.roll(phi_clipped, 1, axis=0)) / (2*dx)
        phi_y = (np.roll(phi_clipped, -1, axis=1) - np.roll(phi_clipped, 1, axis=1)) / (2*dy)
        mag = np.sqrt(phi_x**2 + phi_y**2 + 1e-12)

        nx = phi_x / mag
        ny = phi_y / mag

        # 3) Curvature
        dnx_dx = (np.roll(nx, -1, axis=0) - np.roll(nx, 1, axis=0)) / (2*dx)
        dny_dy = (np.roll(ny, -1, axis=1) - np.roll(ny, 1, axis=1)) / (2*dy)
        kappa = dnx_dx + dny_dy

        # 4) Safe delta function => clamp the argument of cosh
        max_arg = 10.0
        arg = np.clip(phi_clipped / eps, -max_arg, max_arg)
        delta_phi = (1.0/(2*eps))*(1.0/np.cosh(arg))**2

        return kappa, delta_phi, nx, ny

    def momentum_predictor(self, u, v, rho, mu, phi):
        """
        Simplistic momentum step with gravity and surface tension (CSF).
        Omits real convection or viscosity on u, v for brevity.
        """
        dt = self.p.dt
        dx, dy = self.p.dx, self.p.dy
        nx, ny = self.p.nx, self.p.ny

        # (a) Gravity on v
        for i in range(nx):
            for j in range(1, ny):
                rho_vface = 0.5*(rho[i,j-1] + rho[i,j])
                if rho_vface > 1e-12:
                    v[i,j] += dt * self.p.g

        # (b) Surface tension
        sigma = self.p.sigma
        kappa, delta_phi, nx_n, ny_n = self.compute_curvature_and_delta(phi)

        Fx = sigma * kappa * delta_phi * nx_n
        Fy = sigma * kappa * delta_phi * ny_n

        # Distribute to u faces
        for i in range(1, nx):
            for j in range(ny):
                rho_face = 0.5*(rho[i,j] + rho[i-1,j])
                if rho_face > 1e-12:
                    fxc = 0.5*(Fx[i,j] + Fx[i-1,j])
                    u[i,j] += dt*(fxc / rho_face)

        # Distribute to v faces
        for i in range(nx):
            for j in range(1, ny):
                rho_face = 0.5*(rho[i,j] + rho[i,j-1])
                if rho_face > 1e-12:
                    fyc = 0.5*(Fy[i,j] + Fy[i,j-1])
                    v[i,j] += dt*(fyc / rho_face)

        return u, v

    def solve_pressure(self, u, v, rho, max_iter=200, tol=1e-6):
        nx, ny = self.p.nx, self.p.ny
        dx, dy, dt = self.p.dx, self.p.dy, self.p.dt
        p = np.zeros((nx, ny))

        # 1) Divergence of intermediate velocity
        div_star = np.zeros((nx, ny))
        for i in range(1, nx):
            for j in range(ny):
                div_star[i,j] += (u[i,j] - u[i-1,j]) / dx
        for i in range(nx):
            for j in range(1, ny):
                div_star[i,j] += (v[i,j] - v[i,j-1]) / dy

        # 2) Gauss-Seidel for variable-coeff Poisson
        for _ in range(max_iter):
            p_old = p.copy()
            for i in range(1, nx-1):
                for j in range(1, ny-1):
                    rhoE = 0.5*(rho[i,j] + rho[i+1,j])
                    rhoW = 0.5*(rho[i,j] + rho[i-1,j])
                    rhoN = 0.5*(rho[i,j] + rho[i,j+1])
                    rhoS = 0.5*(rho[i,j] + rho[i,j-1])

                    AE = 1./max(rhoE, 1e-12)
                    AW = 1./max(rhoW, 1e-12)
                    AN = 1./max(rhoN, 1e-12)
                    AS = 1./max(rhoS, 1e-12)

                    coeff = (AE+AW)/dx**2 + (AN+AS)/dy**2
                    rhs = div_star[i,j]/dt

                    p_val = ((AE*p[i+1,j] + AW*p[i-1,j]) / dx**2 +
                             (AN*p[i,j+1] + AS*p[i,j-1]) / dy**2 - rhs) / coeff
                    p[i,j] = p_val

            diff = np.max(np.abs(p - p_old))
            if diff < tol:
                break

        # 3) Project velocity
        for i in range(1, nx):
            for j in range(ny):
                rho_face = 0.5*(rho[i,j] + rho[i-1,j])
                dpdx = (p[i,j] - p[i-1,j]) / dx
                if rho_face > 1e-12:
                    u[i,j] -= dt*(dpdx / rho_face)

        for i in range(nx):
            for j in range(1, ny):
                rho_face = 0.5*(rho[i,j] + rho[i,j-1])
                dpdy = (p[i,j] - p[i,j-1]) / dy
                if rho_face > 1e-12:
                    v[i,j] -= dt*(dpdy / rho_face)

        return p

class Simulation:
    def __init__(self, params: SimulationParams = None):
        self.params = params or SimulationParams()
        self.grid = Grid(self.params)
        self.ls_solver = LevelSetSolver()
        self.fluid_solver = FluidSolver(self.params)

    def step(self, step_number):
        # Reinitialize phi (start of step).
        phi0 = self.grid.phi.copy()
        self.grid.phi = self.ls_solver.reinitialize(self.grid.phi, phi0, self.params)

        # Compute fluid properties
        H = 0.5*(1 + np.tanh(self.grid.phi / self.params.epsilon))
        rho = self.params.rho_inner + (self.params.rho_outer - self.params.rho_inner)*H
        mu  = self.params.mu_inner  + (self.params.mu_outer  - self.params.mu_inner)*H

        # Momentum predictor (gravity + surface tension)
        self.grid.u, self.grid.v = self.fluid_solver.momentum_predictor(
            self.grid.u, self.grid.v, rho, mu, self.grid.phi
        )

        # Pressure projection
        p_new = self.fluid_solver.solve_pressure(self.grid.u, self.grid.v, rho)
        self.grid.pressure = p_new

        # Advect phi with updated velocities
        self.grid.phi = self.ls_solver.advect(self.grid.phi, self.grid.u, self.grid.v, self.params)

        # Optional reinit after advection
        phi1 = self.grid.phi.copy()
        self.grid.phi = self.ls_solver.reinitialize(self.grid.phi, phi1, self.params)

    def run(self, plot_interval=10):
        for n in range(self.params.nsteps):
            self.step(n)
            if n % plot_interval == 0:
                self.visualize(n)

        plt.show()

    def visualize(self, step):
        plt.clf()
        # 1) Plot the interface
        plt.contour(
            self.grid.xc,
            self.grid.yc,
            self.grid.phi.T,
            levels=[0],
            colors='red'
        )
        
        # 2) Convert staggered velocities to cell-centered
        uc = 0.5*(self.grid.u[:-1,:] + self.grid.u[1:,:])
        vc = 0.5*(self.grid.v[:,:-1] + self.grid.v[:,1:])

        # 3) Subsample (skip) to reduce clutter
        skip = 8
        Xsub = self.grid.Xc[::skip, ::skip]
        Ysub = self.grid.Yc[::skip, ::skip]
        Usub = uc[::skip, ::skip]
        Vsub = vc[::skip, ::skip]

        # 4) Quiver with a scaling factor to shrink arrow lengths
        plt.quiver(
            Xsub, Ysub,
            Usub, Vsub,
            color='blue',
            scale=30,         # bigger => shorter arrows
            scale_units='xy'  # interpret scale in data units
        )

        plt.title(f"Step {step}, t={step*self.params.dt:.4f}")
        plt.pause(0.01)

if __name__ == "__main__":
    sim = Simulation()
    sim.run(plot_interval=10)