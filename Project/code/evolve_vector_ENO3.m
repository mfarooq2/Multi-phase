function [delta] = evolve_vector_ENO3(phi, dx, dy, u_ext, v_ext)
%
% Finds the amount of evolution under a vector field
% based force and using 3rd order accurate ENO scheme
%
% User m functions called: upwind_ENO3.m
% 
% Author: Baris Sumengen  sumengen@ece.ucsb.edu
% 
% slightly modified by Maged Ismail 04/26/07

delta = zeros(size(phi)+6);
data_ext = zeros(size(phi)+6);
data_ext(4:end-3,4:end-3) = phi;
% first scan the rows
for i=1:size(phi,1)
    delta(i+3,:) = delta(i+3,:) + upwind_ENO3(data_ext(i+3,:), u_ext, dx);
end
% then scan the columns
for j=1:size(phi,2)
    delta(:,j+3) = delta(:,j+3) + upwind_ENO3(data_ext(:,j+3), v_ext, dy);
end
delta = delta(4:end-3,4:end-3);
% end of evolve_vector_ENO3.m