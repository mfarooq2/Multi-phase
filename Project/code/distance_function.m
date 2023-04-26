% A simple distance function evaluator from a point with coordinates(x,y)
% to a given set of points S.
function dmin = distance_function (x,y,S)
dmin = inf;
for i=1:size(S,1)
    dx = abs(S(i,1)-x);
    dy = abs(S(i,2)-y);
    d = sqrt(dx^2+dy^2);
    if d<dmin 
        dmin=d;
    end
end  