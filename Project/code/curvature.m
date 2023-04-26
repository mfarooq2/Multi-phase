function kappa= curvature(phi,h)
% Calculates the curvature of the function phi(x,y)
% where h=dx=dy is the spatial grid size
%
% Maged Ismail 04/25/07

invhh=1/h; [Nx,Ny]=size(phi);

phibc(2:Nx+1,2:Ny+1) = phi; % Copy phi into phibc
phibc(   1,2:Ny+1)   = phi(Nx, :); % Periodic bc
phibc(Nx+2,2:Ny+1)   = phi( 1, :); % Periodic bc
phibc(2:Nx+1,   1)   = -1; % Dirichlet  bc
phibc(2:Nx+1,Ny+2)   = 1;  % Dirichlet  bc

phi=phibc; [Nx,Ny]=size(phi);
term1=zeros(size(phi));term2=term1; term3=term1; term4=term1;
denom1=zeros(size(phi)); denom2=denom1; denom3=denom1; denom4=denom1;

for i=2:Nx-1
    for j=2:Ny-1
        denom1(i,j)=sqrt((phi(i+1,j)-phi(i,j))^2+(1/16)*...
            (phi(i+1,j+1)+phi(i,j+1)-phi(i+1,j-1)-phi(i,j-1))^2);
        denom2(i,j)=sqrt((phi(i,j)-phi(i-1,j))^2+(1/16)*...
            (phi(i-1,j+1)+phi(i,j+1)-phi(i-1,j-1)-phi(i,j-1))^2);      
        denom3(i,j)=sqrt((phi(i,j+1)-phi(i,j))^2+(1/16)*...
            (phi(i+1,j+1)+phi(i+1,j)-phi(i-1,j+1)-phi(i-1,j))^2);
        denom4(i,j)=sqrt((phi(i,j)-phi(i,j-1))^2+(1/16)*...
            (phi(i+1,j-1)+phi(i+1,j)-phi(i-1,j-1)-phi(i-1,j))^2);
        if denom1(i,j)==0
            term1(i,j)=0;
        else term1(i,j)=(phi(i+1,j)-phi(i,j))/denom1(i,j);
        end
        if denom2(i,j)==0
            term2(i,j)=0;
        else term2(i,j)=(phi(i-1,j)-phi(i,j))/denom2(i,j);
        end
        if denom3(i,j)==0
            term3(i,j)=0;
        else term3(i,j)=(phi(i,j+1)-phi(i,j))/denom3(i,j);
        end
        if denom4(i,j)==0
            term4(i,j)=0;
        else term4(i,j)=(phi(i,j-1)-phi(i,j))/denom4(i,j);
        end
    end
end
kappa=invhh*(term1+term2+term3+term4); kappa=kappa(2:end-1,2:end-1);
% end of curvature.m