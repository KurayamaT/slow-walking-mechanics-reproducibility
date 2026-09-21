function floquet_KUO_v2_beta(gam, k_hip, beta_target, opts_in)
%FLOQUET_KUO_V2_BETA  Floquet multiplier analysis for the actively powered
%                     simplest walking model with finite leg mass (beta).
%
%   Extension of floquet_KUO_v2.m with two additions:
%     (1) Mass matrix solve for beta != 0
%     (2) Optional beta-continuation per s (start at beta=0, march to target)
%
%   For beta = 0 the result is identical (within tolerance) to floquet_KUO_v2.
%
%   Usage
%   -----
%   FLOQUET_KUO_V2_BETA()                                  defaults:
%                                                          gam=0,k=-0.08,beta=0
%   FLOQUET_KUO_V2_BETA(GAM, K, BETA)                      single (gam,k,beta)
%   FLOQUET_KUO_V2_BETA(GAM, K, BETA, opts)                tune internals
%
%   opts (struct, all fields optional):
%     .use_continuation  true/false (default true if beta>0)
%     .beta_step         step size for continuation (default 0.01)
%     .s_range           override default speed grid
%     .quiet             suppress per-s diagnostics (default false)
%
%   Output
%   ------
%   CSV: floquet_gam{gam}_k{k}_b{beta}.csv  (16 columns, same schema as v2)
%
%   Implementation notes
%   --------------------
%   1) State y = [theta, theta_dot, phi, phi_dot] (Kuo convention).
%      Collision surface: phi - 2*theta = 0  with rising direction.
%   2) For beta = 0 the swing-leg EOM reduces to an explicit form (Eq.2 of
%      Garcia 1998 plus hip spring k*phi).
%   3) For beta > 0 the 2x2 mass matrix is solved at each step. The matrix
%      is derived from Garcia M_G via theta2_G = pi - phi.
%   4) Collision law: theta_dot_post = (cos(2a)*theta_dot_pre + sin(2a)*P)
%      / (1 + beta*sin^2(2a))                                (Garcia 1998)
%   5) Post-collision phi_dot_post = (1-cos(2a))*theta_dot_post is a
%      kinematic identity (rigid massless rod, foot at rest pre-collision
%      => zero angular momentum about hip), independent of beta.

if nargin < 1, gam = 0; end
if nargin < 2, k_hip = -0.08; end
if nargin < 3, beta_target = 0; end
if nargin < 4, opts_in = struct(); end

if ~isfield(opts_in, 'use_continuation'), opts_in.use_continuation = (beta_target > 0); end
if ~isfield(opts_in, 'beta_step'),        opts_in.beta_step = 0.01; end
if ~isfield(opts_in, 's_range')
    opts_in.s_range = [linspace(0.01, 0.05, 500), linspace(0.05, 0.80, 2000)];
end
if ~isfield(opts_in, 'quiet'),            opts_in.quiet = false; end

per = 5;
s_range = opts_in.s_range;
N = length(s_range);

% beta sequence used for continuation (last element is beta_target)
if opts_in.use_continuation && beta_target > 0
    n_steps  = max(1, ceil(beta_target / opts_in.beta_step));
    beta_seq = linspace(0, beta_target, n_steps + 1);
else
    beta_seq = beta_target;
end

speeds_out = NaN(1, N);
s_out      = NaN(1, N);
lam_out    = complex(NaN(N, 3));
z_fp_out   = NaN(N, 3);
T_out      = NaN(1, N);
valid_out  = false(1, N);
trJ_out    = NaN(1, N);

opts_nr   = odeset('RelTol',1e-8, 'AbsTol',1e-10, 'Refine',4, ...
                   'Events',@collision_with_guard);
opts_fine = odeset('RelTol',1e-12,'AbsTol',1e-14, 'Refine',4, ...
                   'Events',@collision_with_guard);

if ~opts_in.quiet
    fprintf('============================================================\n');
    fprintf(' Floquet Analysis: Kuo (2002) with finite leg mass\n');
    fprintf(' gamma = %.4f,  k_hip = %.4f,  beta = %.4f\n', gam, k_hip, beta_target);
    fprintf(' continuation steps: %d  (Delta beta = %.4f)\n', ...
        length(beta_seq)-1, opts_in.beta_step);
    fprintf(' %d step lengths in [%.3f, %.3f]\n', N, s_range(1), s_range(end));
    fprintf('============================================================\n');
end
tic;

pool = gcp('nocreate');
if isempty(pool), pool = parpool; end
if ~opts_in.quiet, fprintf(' Using %d workers.\n\n', pool.NumWorkers); end

parfor idx = 1:N
    s = s_range(idx);
    alpha = asin(0.5*s);
    omega = -1.04*alpha;
    P = -omega*tan(alpha);
    z_init = [alpha; omega; (1-cos(2*alpha))*omega];

    z_fp = z_init;
    T_stride = NaN;
    converged_at_target = false;

    % --- beta-continuation: march beta_seq one step at a time ---
    for bi = 1:length(beta_seq)
        beta = beta_seq(bi);
        [z_fp_new, T_new, conv] = find_fixed_point(z_fp, gam, k_hip, beta, P, per, opts_nr);
        if ~conv
            % fall back to the standard initial guess once
            [z_fp_new, T_new, conv] = find_fixed_point(z_init, gam, k_hip, beta, P, per, opts_nr);
            if ~conv
                converged_at_target = false;
                break;
            end
        end
        z_fp = z_fp_new;
        T_stride = T_new;
        if bi == length(beta_seq)
            converged_at_target = true;
        end
    end

    if ~converged_at_target, continue; end

    % --- Re-converge at target beta with TIGHT tolerances ---
    [z_fp, T_stride, conv] = find_fixed_point(z_fp, gam, k_hip, beta_target, P, per, opts_fine);
    if ~conv, continue; end

    % Verify residual
    [Sz, ~, ok] = step_map(z_fp, gam, k_hip, beta_target, P, per, opts_fine);
    if ~ok || norm(Sz - z_fp) > 1e-9, continue; end

    % --- Jacobian (3x3) at tight tolerances ---
    delta = 1e-7;
    J = zeros(3, 3);
    ok_all = true;
    for j = 1:3
        zp = z_fp; zp(j) = zp(j) + delta;
        zm = z_fp; zm(j) = zm(j) - delta;
        [Sp, ~, ok1] = step_map(zp, gam, k_hip, beta_target, P, per, opts_fine);
        [Sm, ~, ok2] = step_map(zm, gam, k_hip, beta_target, P, per, opts_fine);
        if ~ok1 || ~ok2, ok_all = false; break; end
        J(:,j) = (Sp - Sm) / (2*delta);
    end
    if ~ok_all, continue; end

    lambda = eig(J);
    [~, si] = sort(abs(lambda), 'descend');
    lambda = lambda(si);

    speeds_out(idx) = s / T_stride;
    s_out(idx)      = s;
    lam_out(idx,:)  = lambda.';
    z_fp_out(idx,:) = z_fp.';
    T_out(idx)      = T_stride;
    valid_out(idx)  = true;
    trJ_out(idx)    = real(trace(J));
end

elapsed = toc;
n_conv = sum(valid_out);
if ~opts_in.quiet
    fprintf('Elapsed: %.1f sec.  Converged: %d / %d\n', elapsed, n_conv, N);
end

% Filter & sort
mask     = valid_out;
speeds   = speeds_out(mask);
s_vals   = s_out(mask);
lam_all  = lam_out(mask, :);
T_vals   = T_out(mask);
trJ_vals = trJ_out(mask);
n_found  = sum(mask);

if n_found == 0
    warning('No periodic orbits found for beta=%.4f. CSV will be empty.', beta_target);
end

[speeds, si] = sort(speeds);
s_vals   = s_vals(si);
lam_all  = lam_all(si, :);
T_vals   = T_vals(si);
trJ_vals = trJ_vals(si);

lam_max = max(abs(lam_all), [], 2);
N_half  = -log(2) ./ log(lam_max);
stable  = lam_max < 1;

% =====================================================================
% CSV EXPORT
% =====================================================================
csv_name = sprintf('floquet_gam%.3f_k%.3f_b%.3f.csv', gam, k_hip, beta_target);
csv_name = strrep(csv_name, '-', 'm');
fid = fopen(csv_name, 'w');
fprintf(fid, ['s,v,T_stride,' ...
    'lam1_re,lam1_im,lam2_re,lam2_im,lam3_re,lam3_im,' ...
    'abs_lam1,abs_lam2,abs_lam3,lam_max,N_half,trace_J,stable\n']);
for i = 1:n_found
    la = lam_all(i,:);
    fprintf(fid, ['%.6f,%.6f,%.6f,' ...
        '%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,' ...
        '%.8f,%.8f,%.8f,%.8f,%.4f,%.6f,%d\n'], ...
        s_vals(i), speeds(i), T_vals(i), ...
        real(la(1)),imag(la(1)), real(la(2)),imag(la(2)), ...
        real(la(3)),imag(la(3)), ...
        abs(la(1)), abs(la(2)), abs(la(3)), ...
        lam_max(i), N_half(i), trJ_vals(i), stable(i));
end
fclose(fid);
if ~opts_in.quiet
    fprintf('Exported %d orbits to %s\n', n_found, csv_name);
end

end  % main function


% =========================================================================
% SUBFUNCTIONS
% =========================================================================

function [z_new, T_stride, ok] = step_map(z, gam, k, beta, P, per, opts)
    z_new = []; T_stride = []; ok = false;
    if abs(z(1)) > pi/3, return; end

    y0 = [z(1); z(2); 2*z(1); z(3)];

    dt = 0.005;
    opts_noevent = odeset('RelTol',1e-12, 'AbsTol',1e-14);
    [~, yout1] = ode45(@(t,y)eom_beta(t,y,gam,k,beta), [0 dt], y0, opts_noevent);
    y_dep = yout1(end,:).';

    [~, ~, te, ye, ie] = ode45(@(t,y)eom_beta(t,y,gam,k,beta), ...
        [dt, per], y_dep, opts);

    if isempty(ie), return; end
    idx_coll = find(ie == 1);
    if isempty(idx_coll), return; end
    ic = idx_coll(1);

    T_stride = te(ic);
    yc = ye(ic,:);

    c2  = cos(2*yc(1));
    s2  = sin(2*yc(1));
    factor = 1 + beta*s2^2;
    theta_dot_post = (c2*yc(2) + s2*P) / factor;
    phi_dot_post   = (1 - c2) * theta_dot_post;

    z_new = [-yc(1); theta_dot_post; phi_dot_post];
    ok = true;
end


function ydot = eom_beta(~, y, gam, k, beta)
    th  = y(1); thd = y(2);
    ph  = y(3); phd = y(4);

    if beta == 0
        F = k * ph;
        ydot = [thd;
                sin(th - gam);
                phd;
                sin(th-gam) + sin(ph)*(thd^2 - cos(th-gam)) + F];
        return;
    end

    cph = cos(ph); sph = sin(ph);
    s1g = sin(th - gam);
    s12 = sin(gam + ph - th);

    M11 =  1 + 2*beta*(1-cph);
    M12 = -beta*(1-cph);
    M21 =  1 - cph;
    M22 = -1;

    rhs1 = -beta*sph*phd*(2*thd - phd) + (beta*s12 + s1g*(1+beta));
    rhs2 = -sph*thd^2 + s12 - k*ph;

    detM = M11*M22 - M12*M21;
    th_dd = (M22*rhs1 - M12*rhs2) / detM;
    ph_dd = (M11*rhs2 - M21*rhs1) / detM;

    ydot = [thd; th_dd; phd; ph_dd];
end


function [val, ist, dir] = collision_with_guard(~, y) %#ok<INUSL>
    val = [y(3) - 2*y(1);
           pi/2 - abs(y(1))];
    ist = [1; 1];
    dir = [1; 0];
end


function [z_fp, T_stride, converged] = find_fixed_point(z0, gam, k, beta, P, per, opts)
    converged = false; T_stride = []; z_fp = z0;
    delta_nr = 1e-7; max_iter = 20; tol = 1e-12;

    [~, ~, ok] = step_map(z_fp, gam, k, beta, P, per, opts);
    if ~ok, return; end

    for iter = 1:max_iter
        [Sz, T, ok] = step_map(z_fp, gam, k, beta, P, per, opts);
        if ~ok, return; end
        T_stride = T;
        res = Sz - z_fp;
        if norm(res) < tol, converged = true; return; end

        Jg = zeros(3);
        for j = 1:3
            zp = z_fp; zp(j) = zp(j) + delta_nr;
            zm = z_fp; zm(j) = zm(j) - delta_nr;
            [Sp,~,ok1] = step_map(zp, gam, k, beta, P, per, opts);
            [Sm,~,ok2] = step_map(zm, gam, k, beta, P, per, opts);
            if ~ok1 || ~ok2, return; end
            Jg(:,j) = (Sp - Sm) / (2*delta_nr);
        end
        Jg = Jg - eye(3);
        if rcond(Jg) < 1e-14, return; end
        z_fp = z_fp + 0.8 * (-Jg \ res);
    end

    [Sz, T, ok] = step_map(z_fp, gam, k, beta, P, per, opts);
    if ok && norm(Sz - z_fp) < 1e-9
        converged = true; T_stride = T;
    end
end
