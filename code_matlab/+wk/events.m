function [value, isterminal, direction] = events(~, y)
%EVENTS  Heel-strike (phi = 2 theta, increasing crossing) and a divergence guard.
%   Both are terminal; the caller checks which one fired.
value      = [y(3) - 2.0 * y(1);  pi / 2.0 - abs(y(1))];
isterminal = [1; 1];
direction  = [1; 0];
end
