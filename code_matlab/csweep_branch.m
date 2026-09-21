function csweep_branch(c_val)
%CSWEEP_BRANCH  Trace one push-off-coefficient branch by seeded continuation.
%   P(s) = c * alpha * tan(alpha), k_hip = -0.16.  The anchor gait at s = 0.40
%   is reached by continuation in c from 1.04 (0.01 steps), then the branch is
%   swept down to s = 0.010 and up to s = 0.800 at 0.001 spacing, each gait
%   seeded with the previous post-impact state (port of the trace() function of
%   code/run_csweep_extended.py).  Saves data_matlab/branches/csweep_c<c>.mat
%   with variables c, s, v, lam_max, T, Z (3 x n post-impact states), elapsed.
%   Usage (one branch per MATLAB session):  csweep_branch(0.80)
C_MAIN = 1.04;
k = wk.constants();
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'data_matlab', 'branches');
if ~exist(outdir, 'dir'), mkdir(outdir); end
out = fullfile(outdir, sprintf('csweep_c%.2f.mat', c_val));

tic;
ref = solve_at(0.40, C_MAIN, [], k);       % geometric guess converges at c = 1.04
assert(~isempty(ref), 'reference gait at s = 0.40, c = 1.04 did not converge');

% anchor at s = 0.40 for this c, by continuation in c from 1.04
if abs(c_val - C_MAIN) < 1e-9
    anchor = ref;
else
    n = round(abs(c_val - C_MAIN) / 0.01);
    cgrid = linspace(C_MAIN, c_val, n + 1); cgrid = cgrid(2:end);
    seed = ref.z; r = ref;
    for ct = cgrid
        r = solve_at(0.40, ct, seed, k);
        if isempty(r), error('c-continuation failed near c=%.3f', ct); end
        seed = r.z;
    end
    anchor = r;
end

rows = {anchor};
sd = anchor.z;
for i = 0:389                                   % 0.399, 0.398, ..., 0.010
    s = 0.399 - 0.001 * i;
    r = solve_at(s, c_val, sd, k);
    if isempty(r), fprintf('  down: ended at s=%.3f\n', s + 0.001); break; end
    sd = r.z; rows{end+1} = r; %#ok<AGROW>
end
su = anchor.z;
for i = 0:399                                   % 0.401, ..., 0.800
    s = 0.401 + 0.001 * i;
    r = solve_at(s, c_val, su, k);
    if isempty(r), fprintf('  up: ended at s=%.3f\n', s - 0.001); break; end
    su = r.z; rows{end+1} = r; %#ok<AGROW>
end
v = cellfun(@(r) r.v, rows);
[~, order] = sort(v); rows = rows(order);        % stable sort on v, as Python
c = c_val; %#ok<NASGU>
s = cellfun(@(r) r.s, rows);
v = cellfun(@(r) r.v, rows);
lam_max = cellfun(@(r) r.lam_max, rows);
T = cellfun(@(r) r.T, rows);
Z = cell2mat(cellfun(@(r) r.z(:), rows, 'UniformOutput', false));
elapsed = toc;
save(out, 'c', 's', 'v', 'lam_max', 'T', 'Z', 'elapsed');
[~, i_lo] = min(v); [~, i_hi] = max(v);
nh = -log(2.0) ./ log(min(max(lam_max, 1e-12), 0.9999999));
fprintf('c=%.2f  n=%d  v_lo=%.4f  Nh_lo=%.1f  v_hi=%.4f  Nh_hi=%.1f  (%.1f s)\nsaved: %s\n', ...
        c_val, numel(v), v(i_lo), nh(i_lo), v(i_hi), nh(i_hi), elapsed, out);
end

function r = solve_at(s, c_val, seed, k)
    alpha = asin(0.5 * s);
    P = c_val * alpha * tan(alpha);
    if isempty(seed)
        z0 = [alpha; -c_val * alpha; (1 - cos(2 * alpha)) * (-c_val * alpha)];
    else
        z0 = seed(:);
    end
    [z_fp, T, ok] = wk.find_fixed_point(z0, k.GAM, k.KHIP, P, k.RTOL, k.ATOL, 1e-12, 20, 1e-7);
    if ~ok, r = []; return; end
    J = wk.jacobian_3d(z_fp, k.GAM, k.KHIP, P, k.RTOL, k.ATOL, k.DELTA);
    if isempty(J), r = []; return; end
    lam = wk.sorted_eigs(J);
    r = struct('s', s, 'v', s / T, 'T', T, 'lam_max', max(abs(lam)), 'z', z_fp);
end
