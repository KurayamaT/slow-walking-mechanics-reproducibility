function c = constants()
%CONSTANTS  Branch and solver constants shared by every wk.* function.
%   Faithful to code/revision_numerics.py and to others/code/floquet/floquet_KUO_v2.m.
c.GAM       = 0.0;     % level ground
c.KHIP      = -0.16;   % reference hip-spring coefficient
c.PER       = 5.0;     % maximum step duration
c.DT_DEPART = 0.005;   % phase-1 departure from the collision surface
c.RTOL      = 1e-12;   % ODE tolerances for the fixed point and the Jacobian
c.ATOL      = 1e-14;
c.DELTA     = 1e-7;    % central-difference step for the Jacobian
end
