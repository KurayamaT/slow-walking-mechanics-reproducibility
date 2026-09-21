function dy = eom(~, y, gam, k)
%EOM  Swing-phase equations of motion of the actively powered simplest walker.
%   y = [theta; theta_dot; phi; phi_dot]; gam = ground slope; k = hip-spring coefficient.
th = y(1); thd = y(2); ph = y(3); phd = y(4);
dy = [thd;
      sin(th - gam);
      phd;
      sin(th - gam) + sin(ph) * (thd * thd - cos(th - gam)) + k * ph];
end
