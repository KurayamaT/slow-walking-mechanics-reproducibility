%VALIDATE_CORE  Reproduce reference gaits of data/master_fixed_khip_extended.csv
%   with the MATLAB core, and report the deviation in |lambda_max| and T.
%   Reference values are the rows of the Python-generated branch.
ref = [ % s       v         T         lam_max      lam1_re      lam1_im      lam2_re      N_half    type(1=real,2=complex)
        0.010  0.002895  3.454566  0.99917734  0.99917734  0.00000000  0.52257225  842.2247  1;
        0.080  0.023156  3.454891  0.94388957  0.94388957  0.00000000  0.55236275   12.0034  1;
        0.142  0.041093  3.455584  0.72087232  0.72032848  0.02799613  0.72032848    2.1178  2;
        0.402  0.116141  3.461298  0.70729548  0.44038400  0.55346980  0.44038400    2.0015  2;
        0.800  0.231213  3.460015  0.64528882 -0.34677392  0.54419253 -0.34677392    1.5823  2];
fprintf('%7s %12s %12s %10s %9s %9s %8s\n', 's', '|lam|_ref', '|lam|_ml', 'd|lam|', 'dT', 'dN_half', 'type');
worst = 0; tic;
for i = 1:size(ref, 1)
    s = ref(i, 1);
    if s < 0.05
        g = wk.solve_gait_cont(s);            % continuation from s = 0.080, ds = 0.002
    else
        g = wk.solve_gait(s);
    end
    if isempty(g), fprintf('%7.3f  FAILED\n', s); worst = Inf; continue; end
    typ = 'real'; if abs(imag(g.lam(1))) > 1e-6, typ = 'complex'; end
    Nh = -log(2) / log(g.lam_max);
    dl = abs(g.lam_max - ref(i, 4)); dT = abs(g.T - ref(i, 3)); dN = abs(Nh - ref(i, 8));
    worst = max(worst, dl);
    fprintf('%7.3f %12.8f %12.8f %10.2e %9.2e %9.2e %8s  lam1=%+.8f%+.8fi\n', ...
            s, ref(i, 4), g.lam_max, dl, dT, dN, typ, real(g.lam(1)), imag(g.lam(1)));
end
fprintf('\nmax |lambda_max| deviation = %.2e   (%.1f s)\n', worst, toc);
if worst < 1e-7, disp('CORE VALIDATED'); else, disp('CORE MISMATCH'); end
