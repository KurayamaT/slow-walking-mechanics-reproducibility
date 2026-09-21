function g = solve_gait_cont(q_target, q_start, ds, gam, k, rtol, atol, delta)
%SOLVE_GAIT_CONT  Reach q_target by seeded continuation from q_start in steps of ds.
%   q is the branch parameter, not the realised step length (see wk.solve_gait).
%   Returns the last gait reached (the target, or the last solvable gait
%   before a turning point), or [] if the start cannot be solved.
c = wk.constants();
if nargin < 2 || isempty(q_start), q_start = 0.080; end
if nargin < 3 || isempty(ds),      ds      = 0.002; end
if nargin < 4 || isempty(gam),     gam     = c.GAM;   end
if nargin < 5 || isempty(k),       k       = c.KHIP;  end
if nargin < 6 || isempty(rtol),    rtol    = c.RTOL;  end
if nargin < 7 || isempty(atol),    atol    = c.ATOL;  end
if nargin < 8 || isempty(delta),   delta   = c.DELTA; end

g = wk.solve_gait(q_start, gam, k, rtol, atol, delta);
if isempty(g), return; end
q = q_start;
if q_target < q_start, direction = -1.0; else, direction = 1.0; end
while (direction < 0 && q > q_target + 1e-9) || (direction > 0 && q < q_target - 1e-9)
    if direction < 0, q = max(q_target, q - ds); else, q = min(q_target, q + ds); end
    gnew = wk.solve_gait(q, gam, k, rtol, atol, delta, g.z_fp);
    if isempty(gnew), return; end           % fold reached before the target
    g = gnew;
end
end
