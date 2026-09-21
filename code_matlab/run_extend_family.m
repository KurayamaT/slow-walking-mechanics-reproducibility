%RUN_EXTEND_FAMILY  The fixed-k_hip branch (c = 1.04) by seeded continuation at
%   0.001 spacing over the BRANCH PARAMETER q in [0.01, 0.80]: anchored at
%   q = 0.40, swept down to 0.010 and up to 0.800, each gait seeded with the
%   previous post-impact state. q prescribes the push-off; the reported step
%   length is the realised s = 2 sin(alpha_h) with alpha_h = z_fp(1).
%   Writes data_matlab/master_fixed_khip_extended.csv with the same schema as
%   the Python-generated data/master_fixed_khip_extended.csv it replaces.
C = 1.04;
c = wk.constants();
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'data_matlab');
if ~exist(outdir, 'dir'), mkdir(outdir); end
out = fullfile(outdir, 'master_fixed_khip_extended.csv');

tic;
anchor = solve_at(0.40, [], C, c);
assert(~isempty(anchor), 'anchor gait at s = 0.40 did not converge');
rows = {};
sd = anchor.z;
for i = 0:390                                   % 0.400, 0.399, ..., 0.010
    q = 0.400 - 0.001 * i;
    r = solve_at(q, sd, C, c);
    if isempty(r), fprintf('  down: ended at q=%.3f\n', q + 0.001); break; end
    sd = r.z; rows{end+1} = r; %#ok<SAGROW>
end
su = anchor.z;
for i = 0:399                                   % 0.401, ..., 0.800
    q = 0.401 + 0.001 * i;
    r = solve_at(q, su, C, c);
    if isempty(r), fprintf('  up: ended at q=%.3f\n', q - 0.001); break; end
    su = r.z; rows{end+1} = r; %#ok<SAGROW>
end
q_all = cellfun(@(r) r.q, rows);
v_all = cellfun(@(r) r.v, rows);
[~, order] = sort(q_all);
rows = rows(order);

fh = fopen(out, 'w');
fprintf(fh, 'q,alpha_p,P,alpha_h,s,v,T,cos2alpha_h,lam_max,recovery_margin,N_half,lam1_re,lam1_im,lam2_re,lam2_im,eig_type,stable\n');
for i = 1:numel(rows)
    r = rows{i}; lm = r.lam_max;
    if lm > 0 && lm < 1, nh = -log(2.0) / log(lm); else, nh = NaN; end
    if abs(imag(r.l1)) > 1e-7, etype = 'complex'; else, etype = 'real'; end
    fprintf(fh, '%.3f,%.10g,%.10g,%.10g,%.10g,%.10g,%.6f,%.10g,%.8f,%.8f,%.4f,%.8f,%.8f,%.8f,%.8f,%s,%d\n', ...
            r.q, r.alpha_p, r.P, r.alpha_h, r.s, r.v, r.T, r.cos2a_h, lm, 1 - lm, nh, ...
            real(r.l1), imag(r.l1), real(r.l2), imag(r.l2), etype, double(lm < 1));
end
fclose(fh);
fprintf('rows=%d  v in [%.4f, %.4f]  (%.1f s)\nsaved: %s\n', numel(rows), min(v_all), max(v_all), toc, out);

function r = solve_at(q, seed, C, c)
    % q prescribes the push-off; alpha_h = z_fp(1) is what the gait realises.
    alpha_p = asin(0.5 * q);
    P = C * alpha_p * tan(alpha_p);
    if isempty(seed)
        z0 = [alpha_p; -C * alpha_p; (1 - cos(2 * alpha_p)) * (-C * alpha_p)];
    else
        z0 = seed(:);
    end
    [z_fp, T, ok] = wk.find_fixed_point(z0, c.GAM, c.KHIP, P, c.RTOL, c.ATOL, 1e-12, 20, 1e-7);
    if ~ok, r = []; return; end
    J = wk.jacobian_3d(z_fp, c.GAM, c.KHIP, P, c.RTOL, c.ATOL, c.DELTA);
    if isempty(J), r = []; return; end
    lam = wk.sorted_eigs(J);
    alpha_h = z_fp(1);
    s = 2.0 * sin(alpha_h);
    r = struct('q', q, 'alpha_p', alpha_p, 'P', P, 'alpha_h', alpha_h, ...
               's', s, 'v', s / T, 'T', T, ...
               'cos2a_h', cos(2 * alpha_h), 'lam_max', abs(lam(1)), ...
               'l1', lam(1), 'l2', lam(2), 'z', z_fp);
end
