function [normphi] = normcd(phi,h)
% Calculates the norm of the advection term |grad(phi(x,y))| using 
% a central difference scheme
% where h=dx=dy is the spatial grid size
%
% Maged Ismail 04/25/07

normphi=zeros(size(phi));
invhh=1/h;
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
        normphi(i,j)=sqrt(((phi(i+1,j)-phi(i-1,j))^2)+((phi(i,j+1)-phi(i,j-1))^2));
    end
end

normphi=0.5*invhh*normphi;
normphi=normphi(2:end-1,2:end-1);
% end of normcd.m