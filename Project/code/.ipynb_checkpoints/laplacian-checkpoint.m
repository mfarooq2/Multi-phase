function d2phi = laplacian(phi,h)
% Calculates the lapacian of the function phi(x,y) using a 9-point 
% finite-difference stencil
% where h=dx=dy is the spatial grid size
%
% Maged Ismail 04/25/07

d2phi=zeros(size(phi));
invh2=1/h^2;
[Nx,Ny]=size(phi);

phibc(2:Nx+1,2:Ny+1) = phi; % Copy phi into phibc
phibc(   1,2:Ny+1)   = phi(Nx, :); % Periodic bc
phibc(Nx+2,2:Ny+1)   = phi( 1, :); % Periodic bc
phibc(2:Nx+1,   1)   = -1; % Dirichlet  bc
phibc(2:Nx+1,Ny+2)   = 1;  % Dirichlet  bc

phi=phibc;
[Nx,Ny]=size(phi);

for i=2:Nx-1
    for j=2:Ny-1
        d2phi(i,j)=2*(phi(i+1,j)+phi(i,j+1)+phi(i-1,j)+phi(i,j-1)-4*phi(i,j))...
            +0.5*(phi(i+1,j+1)+phi(i+1,j-1)+phi(i-1,j+1)+phi(i-1,j-1)-4*phi(i,j));
    end
end

d2phi=(1/3)*invh2*d2phi; d2phi=d2phi(2:end-1,2:end-1);
% end of laplacian.m