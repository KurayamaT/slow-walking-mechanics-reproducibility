function khip_branch(k_val)
%KHIP_BRANCH  Trace one hip-spring-coefficient branch by seeded continuation.
%   c = 1.04.  The anchor gait at q = 0.40 is solved from the geometric guess,
%   or, where that does not converge, reached by continuation in k_hip from
%   -0.16 (the same grid as code/run_khip_sweep_extended.py).  The branch is
%   then swept from q = 0.400 down to 0.010 and from 0.400 up to 0.800 at
%   0.001 spacing, each gait seeded with the previous post-impact state.
%   Saves data_matlab/branches/khip_<k>.mat with variables k_hip, q, s, v,
%   lam_max, N_half, lam1_im, T, Z (3 x n post-impact states), elapsed,
%   anchor_direct (true if the geometric guess converged at q = 0.40).
%   Usage (one branch per MATLAB session):  khip_branch(-0.30)
C = 1.04;
Q_ANCHOR = 0.40;
cst = wk.constants();
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'data_matlab', 'branches');
if ~exist(outdir, 'dir'), mkdir(outdir); end
out = fullfile(outdir, sprintf('khip_%+.3f.mat', k_val));

tic;
% ---- anchor at q = 0.40 for this k_hip -----------------------------------
[z, ~] = solve_fp(Q_ANCHOR, k_val, [], C, cst);
anchor_direct = ~isempty(z);
n_cont = 0;
if ~anchor_direct
    n_cont = max(2, fix(abs(k_val - cst.KHIP) / 0.01) + 1);  % Python: max(2, int(|k-KHIP|/0.01)+1)
    kgrid = linspace(cst.KHIP, k_val, n_cont);
    seed = [];
    for kc = kgrid
        [z, ~] = solve_fp(Q_ANCHOR, kc, seed, C, cst);
        if isempty(z), error('k_hip continuation failed at k=%.3f', kc); end
        seed = z;
    end
end
if anchor_direct
    fprintf('k_hip=%+.3f  anchor from geometric guess  (%.1f s)\n', k_val, toc);
else
    fprintf('k_hip=%+.3f  anchor by continuation in k_hip over %d points  (%.1f s)\n', k_val, n_cont, toc);
end

% ---- sweep down from 0.400 to 0.010, then up from 0.400 to 0.800 ---------
% Python keeps a dict keyed by q, so the q = 0.400 gait of the up sweep
% overwrites the one of the down sweep (both are solved from the anchor seed).
keys = []; RHO = []; IM = []; TT = []; ZZ = zeros(3, 0);
for direction = [-1, +1]
    seed = z;
    if direction < 0, imax = 390; else, imax = 400; end
    for i = 0:imax
        q = round((Q_ANCHOR + direction * 0.001 * i) * 1000) / 1000;
        [zz, T] = solve_fp(q, k_val, seed, C, cst);
        if isempty(zz)
            fprintf('  direction %+d: ended at q=%.3f\n', direction, q - direction * 0.001); break;
        end
        seed = zz;
        P = push_off(q, C);
        J = wk.jacobian_3d(zz, cst.GAM, k_val, P, cst.RTOL, cst.ATOL, cst.DELTA);
        if isempty(J)
            fprintf('  direction %+d: Jacobian failed at q=%.3f\n', direction, q); break;
        end
        lam = wk.sorted_eigs(J);
        j = find(abs(keys - q) < 1e-12, 1);
        if isempty(j), j = numel(keys) + 1; end
        keys(j) = q; RHO(j) = abs(lam(1)); IM(j) = imag(lam(1)); %#ok<AGROW>
        TT(j) = T; ZZ(:, j) = zz(:); %#ok<AGROW>
    end
end
[q, order] = sort(keys);
lam_max = RHO(order); lam1_im = IM(order); T = TT(order); Z = ZZ(:, order);
s = 2 * sin(Z(1, :));
v = s ./ T;
N_half = nan(size(lam_max));
ok = lam_max > 0 & lam_max < 1;
N_half(ok) = -log(2) ./ log(lam_max(ok));
k_hip = k_val;
elapsed = toc;
save(out, 'k_hip', 'q', 's', 'v', 'lam_max', 'N_half', 'lam1_im', 'T', 'Z', 'elapsed', 'anchor_direct');
isc = abs(lam1_im) > 1e-6;
j = find(isc(2:end) & ~isc(1:end-1) & ok(2:end) & ok(1:end-1), 1) + 1;
if isempty(j), vstar = NaN; else, vstar = mean(v(j-1:j)); end
fprintf('  k_hip=%+.3f  n=%3d  q_min=%.3f  s_min=%.6f  v_min=%.6f  N_1/2=%8.1f  v*=%.6f  (%.1f s)\nsaved: %s\n', ...
        k_val, numel(q), q(1), s(1), v(1), N_half(1), vstar, elapsed, out);
end

function P = push_off(q, C)
    a = asin(0.5 * q);
    P = C * a * tan(a);
end

function [z, T] = solve_fp(q, k, seed, C, cst)
    P = push_off(q, C);
    a = asin(0.5 * q);
    if isempty(seed)
        z0 = [a; -C * a; (1 - cos(2 * a)) * (-C * a)];
    else
        z0 = seed(:);
    end
    [z, T, ok] = wk.find_fixed_point(z0, cst.GAM, k, P, cst.RTOL, cst.ATOL, 1e-12, 20, 1e-7);
    if ~ok, z = []; T = []; end
end
