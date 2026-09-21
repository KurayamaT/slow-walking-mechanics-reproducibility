function run_beta_realised(mode)
if nargin < 1, mode = 'all'; end
assert(ismember(string(mode), ["all", "fine", "wide"]));
if strcmp(mode, 'wide'), beta_wide; return; end
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'data_matlab');
python_path = fullfile(here, '..', 'data', 'a5_beta_realised.csv');
betas = [0.00, 0.05, 0.10, 0.15];
q_values = 0.040 + (0:12) * 0.004;
started = tic;
results = cell(size(betas));
if license('test', 'Distrib_Computing_Toolbox')
    pool = gcp('nocreate');
    if isempty(pool), parpool('Threads', 4); end
    parfor i = 1:numel(betas)
        results{i} = beta_branch(betas(i), q_values);
    end
else
    for i = 1:numel(betas)
        results{i} = beta_branch(betas(i), q_values);
    end
end
raw = vertcat(results{:});
raw = sortrows(raw, [1, 2]);
names = {'beta', 'q', 'alpha_h', 's_real', 'T', 'lam_max', 'N_half', ...
    'theta_dot', 'phi_dot', 'fixed_point_residual', 'lam_delta1e7', 'N_delta1e7'};
states = array2table(raw, 'VariableNames', names);
assert(height(states) == 52 && all(isfinite(raw), 'all'));
assert(all(states.fixed_point_residual < 1e-12));
output = states(:, 1:7);
python = sortrows(readtable(python_path), {'beta', 'q'});
assert(height(python) == height(output));
assert(max(abs(python.beta-output.beta)) < 1e-12);
assert(max(abs(python.q-output.q)) < 1e-12);
parity = output(:, 1:2);
compared = {'alpha_h', 's_real', 'T', 'lam_max', 'N_half'};
max_abs = struct();
for i = 1:numel(compared)
    name = compared{i};
    differences = output.(name) - python.(name);
    parity.([name '_difference']) = differences;
    max_abs.(name) = max(abs(differences));
end
limits = struct('alpha_h', 1e-8, 's_real', 2e-8, 'T', 1e-7, ...
    'lam_max', 1e-7, 'N_half', 0.005);
for i = 1:numel(compared)
    name = compared{i};
    assert(max_abs.(name) < limits.(name), '%s parity failed', name);
end
common_s = -Inf;
for i = 1:numel(betas)
    rows = abs(output.beta-betas(i)) < 1e-12;
    assert(all(diff(output.s_real(rows)) > 0));
    common_s = max(common_s, min(output.s_real(rows)));
end
common_s = round(common_s + 1e-6, 6);
expected = [51.61; 95.05; 123.72; 142.44];
common_rows = zeros(numel(betas), 7);
for i = 1:numel(betas)
    m = abs(output.beta-betas(i)) < 1e-12;
    nh = interp1(output.s_real(m), output.N_half(m), common_s, 'linear');
    nh_python = interp1(python.s_real(m), python.N_half(m), common_s, 'linear');
    nh_fine = interp1(states.s_real(m), states.N_delta1e7(m), common_s, 'linear');
    common_rows(i, :) = [betas(i), common_s, nh, nh_python, nh-nh_python, ...
        expected(i), nh_fine];
    fprintf('beta=%.2f  common s=%.6f  MATLAB %.10f  Python %.10f  delta %.3g  delta1e-7 %.10f\n', ...
        betas(i), common_s, nh, nh_python, nh-nh_python, nh_fine);
end
assert(common_s == 0.039524);
assert(all(round(common_rows(:, 3), 2) == expected));
common = array2table(common_rows, 'VariableNames', ...
    {'beta', 's_common', 'N_half_matlab', 'N_half_python', 'difference', ...
    'N_half_reported', 'N_half_delta1e7'});
writetable(output, fullfile(outdir, 'a5_beta_realised.csv'));
writetable(states, fullfile(outdir, 'beta_realised_states.csv'));
writetable(parity, fullfile(outdir, 'beta_realised_parity.csv'));
writetable(common, fullfile(outdir, 'beta_realised_common.csv'));
summary = struct('matlab_version', version, 'matlab_release', version('-release'), ...
    'beta', betas, 'q_min', min(q_values), 'q_max', max(q_values), ...
    'q_spacing', 0.004, 'n_rows', height(output), 'relative_tolerance', 1e-12, ...
    'absolute_tolerance', 1e-14, 'fixed_point_tolerance', 1e-12, ...
    'fixed_point_difference_step', 1e-7, 'jacobian_difference_step', 1e-6, ...
    'comparison_jacobian_difference_step', 1e-7, 'max_absolute_difference', max_abs, ...
    'parity_limits', limits, 'max_fixed_point_residual', max(states.fixed_point_residual), ...
    'common_step_length', common_s, 'reported_half_lives_reproduced', true, ...
    'elapsed_seconds', toc(started));
fid = fopen(fullfile(outdir, 'beta_realised_validation.json'), 'w');
assert(fid >= 0);
fprintf(fid, '%s\n', jsonencode(summary, PrettyPrint=true));
fclose(fid);
fprintf('PASS: %d independent MATLAB gaits; max differences %s\n', height(output), jsonencode(max_abs));
fprintf('Saved data_matlab/a5_beta_realised.csv and validation artifacts (%.1f s).\n', toc(started));
if strcmp(mode, 'all'), beta_wide; end
end

function beta_wide
here = fileparts(mfilename('fullpath'));
outdir = fullfile(here, '..', 'data_matlab');
legacy_dir = fullfile(here, '..', 'legacy', 'results', 'csv');
alphas = [linspace(0.02, 0.10, 40), 0.10+(1:44)*(0.35/44)];
betas = [0, 0.05, 0.10, 0.15];
results = cell(size(betas));
reference = readtable(fullfile(outdir, 'beta_realised_states.csv'));
seeds = zeros(3, numel(betas));
for i = 1:numel(betas)
    anchor = reference(abs(reference.beta-betas(i)) < 1e-12 & abs(reference.q-0.088) < 1e-12, :);
    assert(height(anchor) == 1);
    seeds(:, i) = [anchor.alpha_h; anchor.theta_dot; anchor.phi_dot];
end
started = tic;
if license('test', 'Distrib_Computing_Toolbox')
    pool = gcp('nocreate');
    if isempty(pool), parpool('Threads', 4); end
    parfor i = 1:numel(betas)
        results{i} = wide_branch(betas(i), alphas, seeds(:, i));
    end
else
    for i = 1:numel(betas), results{i} = wide_branch(betas(i), alphas, seeds(:, i)); end
end
raw = sortrows(vertcat(results{:}), [1, 2]);
names = {'beta', 'alpha_p', 'q', 'alpha_h', 's_real', 'T', 'v', 'q_over_T', ...
    'lam_max', 'N_half', 'lam1_re', 'lam1_im', 'lam2_re', 'lam2_im', ...
    'lam3_re', 'lam3_im', 'fixed_point_residual'};
wide = array2table(raw, 'VariableNames', names);
parity_rows = [];
branch_rows = [];
for i = 1:numel(betas)
    b = betas(i);
    target = wide(abs(wide.beta-b) < 1e-12, :);
    legacy = readtable(fullfile(legacy_dir, sprintf('floquet_beta%.3f_k-0.160.csv', b)));
    differences = zeros(height(legacy), 6);
    for j = 1:height(legacy)
        [da, k] = min(abs(target.alpha_p-legacy.alpha(j)));
        assert(da <= 5.1e-7);
        differences(j, :) = [b, target.alpha_p(k), da, ...
            target.lam_max(k)-legacy.lam_max(j), ...
            target.q_over_T(k)-legacy.v(j), target.fixed_point_residual(k)];
    end
    parity_rows = [parity_rows; differences]; %#ok<AGROW>
    low = target(target.q <= 0.088+1e-12, :);
    assert(all(low.lam_max > 0 & low.lam_max < 1));
    trend = all(diff(low.s_real) > 0) && all(diff(low.N_half) < 0);
    assert(trend, 'low-speed trend not reproduced at beta=%.2f', b);
    branch_rows(end+1, :) = [b, height(target), height(legacy), ...
        sum(target.lam_max < 1), min(target.q), max(target.q), ...
        min(target.s_real), max(target.s_real), min(target.v), max(target.v), ...
        max(abs(differences(:, 4))), max(abs(differences(:, 5))), ...
        max(target.fixed_point_residual), trend]; %#ok<AGROW>
    fprintf('wide beta=%.2f: %d MATLAB / %d legacy gaits; max rho difference %.3g; low-speed trend %d\n', ...
        b, height(target), height(legacy), max(abs(differences(:, 4))), trend);
end
parity = array2table(parity_rows, 'VariableNames', ...
    {'beta', 'alpha_p', 'legacy_alpha_rounding', 'rho_difference', ...
    'q_over_T_difference', 'fixed_point_residual'});
branches = array2table(branch_rows, 'VariableNames', ...
    {'beta', 'matlab_gaits', 'legacy_gaits', 'stable_gaits', 'q_min', 'q_max', ...
    's_min', 's_max', 'v_min', 'v_max', 'max_rho_difference', ...
    'max_legacy_speed_difference', 'max_fixed_point_residual', 'low_speed_trend'});
writetable(wide, fullfile(outdir, 'beta_wide_sweep.csv'));
writetable(parity, fullfile(outdir, 'beta_wide_parity.csv'));
writetable(branches, fullfile(outdir, 'beta_wide_summary.csv'));
summary = struct('matlab_version', version, 'matlab_release', version('-release'), ...
    'alpha_p_grid', 'linspace(0.02,0.10,40) plus linspace(0.10,0.45,45) excluding first point', ...
    'continuation', 'fixed beta, q decreased and increased from independently reproduced MATLAB fine-branch q=0.088 state; fresh fixed-point solve at every wide-grid point', ...
    'beta', betas, 'n_rows', height(wide), ...
    'q_min', min(wide.q), 'q_max', max(wide.q), 'relative_tolerance', 1e-12, ...
    'absolute_tolerance', 1e-14, 'fixed_point_tolerance', 1e-12, ...
    'jacobian_difference_step', 1e-6, 'legacy_matched_rows', height(parity), ...
    'max_rho_difference', max(abs(parity.rho_difference)), ...
    'max_legacy_speed_difference', max(abs(parity.q_over_T_difference)), ...
    'max_fixed_point_residual', max(wide.fixed_point_residual), ...
    'low_speed_trend_reproduced', all(branches.low_speed_trend), ...
    'elapsed_seconds', toc(started));
fid = fopen(fullfile(outdir, 'beta_wide_validation.json'), 'w');
assert(fid >= 0);
fprintf(fid, '%s\n', jsonencode(summary, PrettyPrint=true));
fclose(fid);
assert(all(wide.fixed_point_residual < 1e-12));
assert(summary.max_rho_difference < 5e-5, 'Legacy rho parity exceeds 5e-5');
assert(summary.max_legacy_speed_difference < 2e-6, 'Legacy q/T parity exceeds 2e-6');
fprintf('PASS: wide finite-mass sweep reproduced in MATLAB (%d gaits, %.1f s).\n', height(wide), toc(started));
end

function rows = wide_branch(beta, alphas, anchor)
rows = zeros(numel(alphas), 17);
for direction = [-1, 1]
    z = anchor;
    if direction < 0
        indices = find(2*sin(alphas) <= 0.088);
        indices = fliplr(indices);
    else
        indices = find(2*sin(alphas) > 0.088);
    end
    for j = indices
        a = alphas(j); P = 1.04*a*tan(a);
        [z, T] = fixed_point(z, beta, P, 80);
        assert(~isempty(z), 'wide fixed point failed alpha_p=%.9f beta=%.2f', a, beta);
        [rho, eigenvalues] = multiplier(z, beta, P, 1e-6);
        nhalf = NaN;
        if rho > 0 && rho < 1, nhalf = -log(2)/log(rho); end
        [mapped, ~] = step_map(z, beta, P);
        q = 2*sin(a); sr = 2*sin(z(1));
        rows(j, :) = [beta, a, q, z(1), sr, T, sr/T, q/T, rho, nhalf, ...
            real(eigenvalues(1)), imag(eigenvalues(1)), real(eigenvalues(2)), ...
            imag(eigenvalues(2)), real(eigenvalues(3)), imag(eigenvalues(3)), norm(mapped-z)];
    end
end
fprintf('Completed wide beta=%.2f: %d periodic gaits.\n', beta, numel(alphas));
end

function rows = beta_branch(beta, q_values)
rows = zeros(numel(q_values), 12);
seed = [];
for i = numel(q_values):-1:1
    q = q_values(i);
    a = asin(q/2);
    omega = -1.04*a;
    P = -omega*tan(a);
    if isempty(seed)
        seed = [a; omega; (1-cos(2*a))*omega];
        if beta > 0
            n = max(1, ceil(beta/0.01));
            for b = linspace(0, beta, n+1)
                [seed, ~] = fixed_point(seed, b, P);
                assert(~isempty(seed), 'beta continuation failed at beta=%.3f', b);
            end
        end
    end
    [z, T] = fixed_point(seed, beta, P);
    assert(~isempty(z), 'fixed point failed at beta=%.2f q=%.3f', beta, q);
    seed = z;
    rho = multiplier(z, beta, P, 1e-6);
    rho_fine = multiplier(z, beta, P, 1e-7);
    [mapped, ~] = step_map(z, beta, P);
    assert(0 < rho && rho < 1 && 0 < rho_fine && rho_fine < 1);
    rows(i, :) = [beta, q, z(1), 2*sin(z(1)), T, rho, -log(2)/log(rho), ...
        z(2), z(3), norm(mapped-z), rho_fine, -log(2)/log(rho_fine)];
end
fprintf('Completed beta=%.2f: %d periodic gaits.\n', beta, numel(q_values));
end

function [rho, eigenvalues] = multiplier(z, beta, P, delta)
J = zeros(3);
for j = 1:3
    zp = z; zm = z;
    zp(j) = zp(j) + delta; zm(j) = zm(j) - delta;
    [Sp, ~] = step_map(zp, beta, P);
    [Sm, ~] = step_map(zm, beta, P);
    assert(~isempty(Sp) && ~isempty(Sm));
    J(:, j) = (Sp-Sm)/(2*delta);
end
eigenvalues = eig(J);
[~, order] = sort(abs(eigenvalues), 'descend');
eigenvalues = eigenvalues(order);
rho = abs(eigenvalues(1));
end

function [z, T] = fixed_point(z, beta, P, max_iterations)
if nargin < 4, max_iterations = 25; end
T = NaN;
for iteration = 1:max_iterations
    [S, T] = step_map(z, beta, P);
    if isempty(S), z = []; return; end
    residual = S-z;
    if norm(residual) < 1e-12, return; end
    Jg = zeros(3);
    for j = 1:3
        zp = z; zm = z;
        zp(j) = zp(j)+1e-7; zm(j) = zm(j)-1e-7;
        [Sp, ~] = step_map(zp, beta, P);
        [Sm, ~] = step_map(zm, beta, P);
        if isempty(Sp) || isempty(Sm), z = []; return; end
        Jg(:, j) = (Sp-Sm)/2e-7;
    end
    z = z+0.8*((Jg-eye(3))\(-residual));
end
z = [];
end

function [znew, T] = step_map(z, beta, P)
znew = []; T = NaN;
if abs(z(1)) > pi/3, return; end
y0 = [z(1); z(2); 2*z(1); z(3)];
options = odeset('RelTol', 1e-12, 'AbsTol', 1e-14);
[~, y] = ode45(@(t, y) eom(t, y, beta), [0, 0.005], y0, options);
options = odeset(options, 'Events', @collision);
[~, ~, te, ye, ie] = ode45(@(t, y) eom(t, y, beta), [0.005, 5], y(end, :).', options);
j = find(ie == 1, 1);
if isempty(j), return; end
T = te(j); yc = ye(j, :);
c2 = cos(2*yc(1)); s2 = sin(2*yc(1));
w = (c2*yc(2)+s2*P)/(1+beta*s2^2);
znew = [-yc(1); w; (1-c2)*w];
end

function dy = eom(~, y, beta)
th = y(1); thd = y(2); ph = y(3); phd = y(4);
k = -0.16;
if beta == 0
    dy = [thd; sin(th); phd; sin(th)+sin(ph)*(thd^2-cos(th))+k*ph];
    return;
end
cp = cos(ph); sp = sin(ph); s12 = sin(ph-th);
M11 = 1+2*beta*(1-cp); M12 = -beta*(1-cp); M21 = 1-cp; M22 = -1;
r1 = -beta*sp*phd*(2*thd-phd)+beta*s12+sin(th)*(1+beta);
r2 = -sp*thd^2+s12-k*ph;
d = M11*M22-M12*M21;
dy = [thd; (M22*r1-M12*r2)/d; phd; (M11*r2-M21*r1)/d];
end

function [value, stop, direction] = collision(~, y)
value = [y(3)-2*y(1); pi/2-abs(y(1))];
stop = [1; 1]; direction = [1; 0];
end
