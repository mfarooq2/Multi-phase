import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import gaussian_filter
from scipy.sparse.linalg import cg

class TwoPhaseFlowSimulator:
    def __init__(self, nx, ny, dx, dy, dt):
        # Grid parameters
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        
        # Physical parameters - scaled for numerical stability
        self.rho1 = 1.0     # Normalized density of fluid 1 (water)
        self.rho2 = 1e-5  # Further reduced density of fluid 2 (air) to enhance buoyancy effects
        self.mu1 = 1e-4   # Further reduced viscosity for fluid 1
        self.mu2 = 1.0e-6   # Further reduced viscosity for fluid 2
        self.sigma = 0.001   # Further reduced surface tension to avoid overly stabilizing the interface
        self.epsilon = 0.005 # Further reduced interface thickness for sharper boundary
        self.mobility = 1.0 # Adjusted mobility for better stability
        self.gravity = 50  # Adjusted gravity for a more gradual effect
        
        # Initialize fields
        self.u = np.zeros((ny, nx))
        self.v = np.zeros((ny, nx))
        self.p = np.zeros((ny, nx))
        self.phi = np.zeros((ny, nx))  # Level set field (phi)
        self.rho = np.zeros((ny, nx))
        self.mu = np.zeros((ny, nx))
        
        # Initialize bubble using level set method
        self._initialize_bubble()
        # Add small perturbation to the vertical velocity to encourage bubble rise
        self.v += 0.01 * (np.random.rand(ny, nx) - 0.5)
        
    def _initialize_bubble(self):
        x = np.linspace(0, self.nx*self.dx, self.nx)
        y = np.linspace(0, self.ny*self.dy, self.ny)
        X, Y = np.meshgrid(x, y)
        
        # Center of the bubble
        x0, y0 = self.nx*self.dx/2, self.ny*self.dy/8
        R = min(self.nx*self.dx, self.ny*self.dy)/10  # Increased initial radius for better buoyancy
        
        # Initialize level set field (phi) for the bubble interface
        distance = np.sqrt((X-x0)**2 + (Y-y0)**2)
        self.phi = distance - R  # Level set initialization (distance from the bubble boundary)
        
        self._update_physical_properties()

    def _update_physical_properties(self):
        # Smoothed Heaviside function for level set
        H = 0.5 * (1 + 2 / np.pi * np.arctan(self.phi / self.epsilon))
        
        # Linear interpolation for density
        self.rho = self.rho1 * (1 - H) + self.rho2 * H
        
        # Logarithmic interpolation for viscosity to handle large contrasts
        log_mu = (1 - H) * np.log(self.mu1) + H * np.log(self.mu2)
        self.mu = np.exp(log_mu)
    
    def _safe_gradient(self, f, spacing, axis):
        grad = np.gradient(f, spacing, axis=axis)
        grad = np.clip(grad, -1e2, 1e2)  # Limit gradient to prevent overflow
        return grad
    
    def _calculate_curvature(self):
        # Calculate curvature for surface tension force using level set method
        phi_x = self._safe_gradient(self.phi, self.dx, 1)
        phi_y = self._safe_gradient(self.phi, self.dy, 0)
        norm_grad_phi = np.sqrt(phi_x**2 + phi_y**2 + 1e-10)  # Add small value to prevent division by zero
        
        phi_xx = self._safe_gradient(phi_x / norm_grad_phi, self.dx, 1)
        phi_yy = self._safe_gradient(phi_y / norm_grad_phi, self.dy, 0)
        
        # More explicit curvature calculation without Gaussian smoothing to maintain sharper interface
        curvature = phi_xx + phi_yy
        curvature = np.clip(curvature, -1e3, 1e3)  # Limit curvature to prevent instability
        return curvature
    
    def _solve_momentum_equations(self):
        # Calculate velocity gradients
        du_dx = self._safe_gradient(self.u, self.dx, 1)
        du_dy = self._safe_gradient(self.u, self.dy, 0)
        dv_dx = self._safe_gradient(self.v, self.dx, 1)
        dv_dy = self._safe_gradient(self.v, self.dy, 0)
        
        # Pressure gradients
        dp_dx = self._safe_gradient(self.p, self.dx, 1)
        dp_dy = self._safe_gradient(self.p, self.dy, 0)
        
        # Viscous terms
        visc_u = (self._safe_gradient(self.mu * du_dx, self.dx, 1) + 
                 self._safe_gradient(self.mu * du_dy, self.dy, 0))
        visc_v = (self._safe_gradient(self.mu * dv_dx, self.dx, 1) + 
                 self._safe_gradient(self.mu * dv_dy, self.dy, 0))
        
        # Buoyancy calculation
        rho_mean = np.mean(self.rho)
        buoyancy = (self.rho - rho_mean) * self.gravity
        
        # Surface tension force using level set curvature
        curvature = self._calculate_curvature()
        surface_tension_x = self.sigma * curvature * self._safe_gradient(self.phi, self.dx, 1)
        surface_tension_y = self.sigma * curvature * self._safe_gradient(self.phi, self.dy, 0)
        
        # Update velocities
        rho_safe = np.clip(self.rho, 1e-3, None)  # Avoid division by very small numbers
        du = self.dt * (
            - (self.u * du_dx + self.v * du_dy)
            - dp_dx / rho_safe
            + visc_u / rho_safe
            + surface_tension_x / rho_safe
        )
        
        dv = self.dt * (
            - (self.u * dv_dx + self.v * dv_dy)
            - dp_dy / rho_safe
            + visc_v / rho_safe
            - buoyancy / rho_safe
            + surface_tension_y / rho_safe
        )
        
        self.u = np.clip(self.u + du, -1e2, 1e2)  # Limit velocity to prevent overflow
        self.v = np.clip(self.v + dv, -1e2, 1e2)  # Limit velocity to prevent overflow
        
    def _pressure_projection(self, max_iter=1000):  # Increased iterations for pressure projection
        # Iterative pressure projection step using Conjugate Gradient method
        div = (self._safe_gradient(self.u, self.dx, 1) + self._safe_gradient(self.v, self.dy, 0))
        div_flat = div.flatten()
        p_flat = self.p.flatten()
        A = np.eye(len(p_flat)) * (-4)  # Simplified representation of Laplace operator
        b = -div_flat
        p_flat, _ = cg(A, b, x0=p_flat, maxiter=max_iter)
        self.p = p_flat.reshape(self.p.shape)
        
    def _reinitialize_level_set(self, reinit_interval=2):  # Reinitialize more frequently
        # Reinitialize the level set function to maintain signed distance property
        if reinit_interval > 0:
            sign_phi = np.sign(self.phi)
            grad_phi = np.sqrt(self._safe_gradient(self.phi, self.dx, 1)**2 + self._safe_gradient(self.phi, self.dy, 0)**2)
            self.phi -= self.dt * sign_phi * (grad_phi - 1)
            # Apply more frequent reinitialization for better accuracy
            self.phi = gaussian_filter(self.phi, sigma=0.05)  # Smoother filtering to maintain stable interface
    
    def _weno_flux(self, stencil):
        # Implementing a more sophisticated WENO reconstruction for a higher-order accurate flux computation
        eps = 1e-6
        beta = [13/12 * (stencil[i-2] - 2*stencil[i-1] + stencil[i])**2 + 
                1/4 * (stencil[i-2] - stencil[i])**2 for i in range(2, 5)]
        
        alpha = [0.1 / (eps + beta[i])**2 for i in range(3)]
        sum_alpha = sum(alpha)
        weights = [alpha[i] / sum_alpha for i in range(3)]
        
        flux = (weights[0] * (stencil[0] + stencil[1]) / 2 +
                weights[1] * (stencil[1] + stencil[2]) / 2 +
                weights[2] * (stencil[2] + stencil[3]) / 2)
        
        return flux
    
    def _weno_advection(self, phi):
        # WENO advection scheme for better interface tracking with higher-order accuracy
        phi_new = np.copy(phi)
        for j in range(3, self.ny - 3):
            for i in range(3, self.nx - 3):
                # Extract stencils for WENO
                stencil = [phi[j, i-3], phi[j, i-2], phi[j, i-1], phi[j, i], phi[j, i+1], phi[j, i+2], phi[j, i+3]]
                
                # Compute flux using the WENO reconstruction
                flux = self._weno_flux(stencil)
                
                # Update phi using flux
                phi_new[j, i] = phi[j, i] - self.dt * flux
        
        return phi_new
    
    def adaptive_time_step(self):
        # Adaptive time-stepping based on CFL condition
        max_u = np.max(np.abs(self.u))
        max_v = np.max(np.abs(self.v))
        max_vel = max(max_u, max_v, 1e-3)  # Avoid division by zero
        cfl = 0.05  # Further reduced CFL condition number for increased stability
        self.dt = cfl * min(self.dx, self.dy) / max_vel
    
    def step(self, iteration):
        """Perform one time step"""
        self.adaptive_time_step()  # Adjust time step adaptively
        self.phi = self._weno_advection(self.phi)  # Use WENO scheme for advecting level set
        self._solve_momentum_equations()
        self._pressure_projection(max_iter=1000)  # Increased iterations for better incompressibility enforcement
        if iteration % 2 == 0:  # Reinitialize every 2 steps for stability
            self._reinitialize_level_set()
        self._update_physical_properties()
        
        # Boundary conditions
        self.u[:, 0] = self.u[:, -1] = 0  # No-slip on side walls
        self.v[:, 0] = self.v[:, -1] = 0  # No-slip on side walls
        self.u[0, :] = self.u[1, :]  # Free-slip on top
        self.u[-1, :] = self.u[-2, :]  # Free-slip on bottom
        self.v[0, :] = 0
        self.v[-1, :] = 0
        
        # Phase field boundary conditions
        self.phi[:, 0] = self.phi[:, 1]
        self.phi[:, -1] = self.phi[:, -2]
        self.phi[0, :] = self.phi[1, :]
        self.phi[-1, :] = self.phi[-2, :]

def run_simulation():
    # Simulation parameters
    nx, ny = 200, 400
    dx = dy = 0.0005
    dt = 1e-7  # Further reduced time step for improved numerical stability
    
    # Initialize simulator
    simulator = TwoPhaseFlowSimulator(nx, ny, dx, dy, dt)
    
    # Set up visualization
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(24, 10))
    
    # Phase field plot
    img1 = ax1.imshow(simulator.phi, cmap='coolwarm', origin='lower',
                    extent=[0, nx*dx, 0, ny*dy])
    ax1.set_title('Level Set Field\n(Blue: Water, Red: Air)')
    plt.colorbar(img1, ax=ax1)
    
    # Velocity magnitude plot
    vel_mag = np.sqrt(np.clip(simulator.u**2 + simulator.v**2, 0, 1e4))  # Clip to prevent overflow in visualization
    img2 = ax2.imshow(vel_mag, cmap='viridis', origin='lower',
                    extent=[0, nx*dx, 0, ny*dy])
    ax2.set_title('Velocity Magnitude')
    plt.colorbar(img2, ax=ax2)
    
    # Advection of level set function plot
    img3 = ax3.imshow(simulator.phi, cmap='plasma', origin='lower',
                    extent=[0, nx*dx, 0, ny*dy])
    ax3.set_title('Advection of Level Set Function')
    plt.colorbar(img3, ax=ax3)
    
    # Pressure field plot
    img4 = ax4.imshow(simulator.p, cmap='inferno', origin='lower',
                    extent=[0, nx*dx, 0, ny*dy])
    ax4.set_title('Pressure Field')
    plt.colorbar(img4, ax=ax4)
    
    # Density field plot
    img5 = ax5.imshow(simulator.rho, cmap='viridis', origin='lower',
                    extent=[0, nx*dx, 0, ny*dy])
    ax5.set_title('Density Field')
    plt.colorbar(img5, ax=ax5)

    def update(frame):
        simulator.step(frame)
        
        img1.set_array(simulator.phi)
        vel_mag = np.sqrt(np.clip(simulator.u**2 + simulator.v**2, 0, 1e4))  # Clip to prevent overflow in visualization
        img2.set_array(vel_mag)
        img3.set_array(simulator.phi)
        img4.set_array(simulator.p)
        img5.set_array(simulator.rho)
        
        # Print maximum velocity for tracking
        max_velocity = np.max(np.sqrt(simulator.u**2 + simulator.v**2))
        print(f"Max velocity at frame {frame}: {max_velocity}")
        
        return img1, img2, img3, img4, img5
    
    anim = FuncAnimation(fig, update, frames=400, interval=50, blit=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()