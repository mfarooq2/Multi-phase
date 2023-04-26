% evolve_vector.m   04/27/2006  Maged Ismail
function evolve_vector

% Interface motion due to external flow fields for
% the diagonal translation of a circle test problem
%
% reproducing Fig. 12 (a) of Sun, Y., and Beckermann, C.,
% "Sharp Interface Tracking Using the Phase-Field Equation,"
% J. Computational Physics, Vol. 220, 2007, pp. 626-653.
%
% Finite-difference method is used to solve the phase-field equation.
% Simple forward Euler method (explicit) is used for time discretization.
% 3rd order HJ ENO scheme is used for calculate the numerical fluxes for
% the hyberbolic term.
%
% User m functions called: curvature.m, normcd.m, laplacian.m and
% evolve_vector_ENO3.m

clear all
clc

% Generate the grid and define the parameters
Nx=11; Ny=Nx;
h=1/(Nx-1);
W=2*h;
bp=0.5;
b=W*bp; 
W2_inv=1/W^2;
xm=0:h:1; ym=0:h:1;
[Y,X]=meshgrid(ym,xm);
xc=.25; yc=.25; r=.15;

% Initialize phi
for i=1:length(X)  
    for j=1:length(Y)
          phi(i,j) = r - sqrt ( ( X(i,j) - xc ).^2 + ( Y(i,j) - yc ).^2 );
    end
end

for i=1:length(xm)
    for j=1:length(ym)
        if phi(i,j)>0.03
            phi(i,j)=1;
        elseif phi(i,j)<-.03
            phi(i,j)=-1;
        end
    end
end

contour(X,Y,phi,[0 0],'b:');
axis([0 1 0 1])
axis square
hold on

dt=0.001; tf=0.5;

% Diagonal translation of the circle
u_ext=1; v_ext=1; 
for t=0:dt:tf
    kappa= curvature(phi,h); normphi = normcd(phi,h); 
    d2phi = laplacian(phi,h); 
    [delta] = evolve_vector_ENO3(phi, h, h, u_ext, v_ext);
    for i=2:Nx-1
        for j=2:Ny-1
            phi(i,j)=phi(i,j)+dt*(b*(d2phi(i,j)+W2_inv*phi(i,j)*...
                (1-phi(i,j)^2)-normphi(i,j)*kappa(i,j))-delta(i,j));                
        end
    end
end

contour(X,Y,phi,[0 0],'b');

% Return the circle to the initial position 
u_ext=-1; v_ext=-1; 
for t=0:dt:tf
    kappa= curvature(phi,h); normphi = normcd(phi,h); 
    d2phi = laplacian(phi,h); 
    [delta] = evolve_vector_ENO3(phi, h, h, u_ext, v_ext);
    for i=2:Nx-1
        for j=2:Ny-1
            phi(i,j)=phi(i,j)+dt*(b*(d2phi(i,j)+W2_inv*phi(i,j)*...
                (1-phi(i,j)^2)-normphi(i,j)*kappa(i,j))-delta(i,j));
        end
    end
end

contour(X,Y,phi,[0 0],'b');
% title('Calculated \phi = 0 contours at half domain translation and after return to the initial position for diagonal translation of a circle using a 160 x160 mesh')
xlabel('x'); ylabel('y');
hold off
% end of evolve_vector.m