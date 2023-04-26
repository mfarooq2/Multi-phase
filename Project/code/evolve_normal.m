% evolve_normal.m   04/25/2006  Maged Ismail
function evolve_normal

% Interface motion with a constant normal speed for the propagating cosine
% curve test problem
%
% reproducing Fig. 6 (a) of Sun, Y., and Beckermann, C.,
% "Sharp Interface Tracking Using the Phase-Field Equation,"
% J. Computational Physics, Vol. 220, 2007, pp. 626-653.
%
% Finite-difference method is used to solve the phase-field equation.
% Simple forward Euler method (explicit) is used for time discretization. 
%
% User m functions called: distance_function.m, curvature.m, normcd.m and
% laplacian.m

clear all
clc

% Generate the grid and define the parameters
nx=101; ny=nx;
h=1/(nx-1);
W=2*h;
bp=0.5;
b=W*bp; 
W2_inv=1/W^2;
xm=-0.1:h:1.1; ym=-0.1:h:1.1;
[Y,X]=meshgrid(ym,xm);
x=-0.1:h:1.1; y=(1+cos(2*pi*(1-x)))/4;
C=[x;y]';   % contour
Nx=length(xm); Ny=length(ym);
Mark = zeros(size(X));
for i=1:length(xm)
    for j=1:length(ym)
        if (Y(i,j))<(((1+cos(2*pi*(1-X(i,j))))/4))
            Mark(i,j)=-1;
        end
    end
end

for i=1:length(xm)
    for j=1:length(ym)
        phi(i,j)=distance_function(X(i,j),Y(i,j),C);
        if Mark(i,j)==-1
            phi(i,j) = -phi(i,j);
        end
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

kappa= curvature(phi,h); normphi=normcd(phi,h); d2phi=laplacian(phi,h);

contour(X,Y,phi,[0 0],'b','linewidth',1);
hold on

dt=0.001; tf=0.1;
t0=0:0.1:0.3; tf=0.1:0.1:0.4;
tic
for p=1:length(tf)
    for t=t0(p):dt:tf(p)
        kappa=curvature(phi,h); normphi=normcd(phi,h); d2phi=laplacian(phi,h);
        for i=2:Nx-1
            for j=2:Ny-1

                phi(i,j)=phi(i,j)+dt*(b*(d2phi(i,j)+W2_inv*phi(i,j)*...
                    (1-phi(i,j)^2)-normphi(i,j)*kappa(i,j))-normphi(i,j));

            end
        end
    end
    contour(X,Y,phi,[0 0],'b','linewidth',1);
end
toc
axis([0 1 -.1 1.1]); xlabel('x'); ylabel('y'); axis square;
% title('Calculated \phi = 0 contours for the propagating cosine curve test problem using a 100 x 120 mesh')
hold off
% end of evolve_normal.m