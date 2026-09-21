%RUN_JACOBIAN_DECOMPOSITION  Where does the approach of |lambda_max| to one come from?
%
%   The return map is R = G o F: F is the swing phase up to heel-strike (swing
%   dynamics plus the dependence of the event time on the state), G is the reset
%   (push-off, collision, leg exchange). Its Jacobian factors as J = B * A with
%   A = dF/dz (numerical) and B = dG/dy (analytic, wk.reset_map). Along the branch
%   the dominant multiplier changes as
%       d lambda/dq = l' (dB/dq A + B dA/dq) r / (l' r)
%   The branch is stepped in q, so every derivative here is with respect to q.
%   Convert with d/ds = (d/dq)/(ds/dq) if a derivative in realised s is wanted.
%   for left/right eigenvectors l, r (first-order perturbation of a simple
%   eigenvalue). The two terms attribute the change to the reset and to the
%   swing/event factor. The reset term is further split by which of its
%   arguments moves along the branch: the collision angle alpha, the pre-impact
%   velocity theta_dot-, and the prescribed push-off P(s).
%
%   Checks built in: B*A against the directly differentiated J; the perturbation
%   derivative against a finite difference of lambda along the branch; the three
%   reset parts against the total reset term. Writes data/jacobian_decomposition.csv.
C = 1.04; c = wk.constants();
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'data_matlab', 'jacobian_decomposition.csv');
h = 0.001;                                   % spacing in q, the branch parameter
tic;

% ---- the branch, s = 0.400 down to 0.010, storing the fixed points ----------
Q = 0.400:-h:0.010; n = numel(Q);   % the grid is in q, not in realised s
Z = zeros(3, n); Pv = zeros(1, n); Tv = zeros(1, n);
seed = [];
for i = 1:n
    g = wk.solve_gait(Q(i), c.GAM, c.KHIP, c.RTOL, c.ATOL, c.DELTA, seed);
    assert(~isempty(g), 'continuation failed at q=%.3f', Q(i));
    Z(:, i) = g.z_fp; Pv(i) = g.P; Tv(i) = g.T; seed = g.z_fp;
end
fprintf('branch: %d gaits (%.0f s)\n', n, toc);

% ---- A, B, J at every gait ----------------------------------------------------
A = cell(1, n); B = cell(1, n); Y = zeros(3, n);
for i = 1:n
    [ym, ~, ok] = wk.flow_to_event(Z(:, i), c.GAM, c.KHIP, c.RTOL, c.ATOL); assert(ok);
    Y(:, i) = ym;
    A{i} = wk.jacobian_flow(Z(:, i), c.GAM, c.KHIP, c.RTOL, c.ATOL, c.DELTA);
    [~, B{i}] = wk.reset_map(ym, Pv(i));
end
fprintf('factors: done (%.0f s)\n', toc);

% ---- decomposition at interior points ----------------------------------------
rows = [];
for i = 2:n-1
    q = Q(i); Ai = A{i}; Bi = B{i}; J = Bi * Ai;
    Jfd = wk.jacobian_3d(Z(:, i), c.GAM, c.KHIP, Pv(i), c.RTOL, c.ATOL, c.DELTA);
    errBA = norm(J - Jfd, 'fro');
    [V, D, W] = eig(J);
    lam = diag(D); [~, k] = max(abs(lam)); lam = lam(k); r = V(:, k); l = W(:, k);
    nrm = l' * r;
    % derivatives along the branch: central differences over neighbouring gaits
    % (S is descending, so i+1 is s-h and i-1 is s+h)
    dA = (A{i-1} - A{i+1}) / (2 * h);
    dB = (B{i-1} - B{i+1}) / (2 * h);
    c_swing = (l' * (Bi * dA) * r) / nrm;
    c_reset = (l' * (dB * Ai) * r) / nrm;
    % reset term split by argument: B = B(alpha, theta_dot-, P)
    ep = 1e-6;
    a0 = Y(1, i); w0 = Y(2, i); P0 = Pv(i);
    da = (Y(1, i-1) - Y(1, i+1)) / (2 * h);
    dw = (Y(2, i-1) - Y(2, i+1)) / (2 * h);
    dP = (Pv(i-1) - Pv(i+1)) / (2 * h);
    [~, Bap] = wk.reset_map([a0 + ep; w0; 0], P0); [~, Bam] = wk.reset_map([a0 - ep; w0; 0], P0);
    [~, Bwp] = wk.reset_map([a0; w0 + ep; 0], P0); [~, Bwm] = wk.reset_map([a0; w0 - ep; 0], P0);
    [~, BPp] = wk.reset_map([a0; w0; 0], P0 + ep); [~, BPm] = wk.reset_map([a0; w0; 0], P0 - ep);
    dB_geom = (Bap - Bam) / (2 * ep) * da;
    dB_vel  = (Bwp - Bwm) / (2 * ep) * dw;
    dB_push = (BPp - BPm) / (2 * ep) * dP;
    c_geom = (l' * (dB_geom * Ai) * r) / nrm;
    c_vel  = (l' * (dB_vel  * Ai) * r) / nrm;
    c_push = (l' * (dB_push * Ai) * r) / nrm;
    % finite-difference derivative of the dominant multiplier itself
    lp = wk.sorted_eigs(B{i-1} * A{i-1}); lm = wk.sorted_eigs(B{i+1} * A{i+1});
    dlam_fd = (lp(1) - lm(1)) / (2 * h);
    eA = wk.sorted_eigs(Ai);
    alpha_h = Z(1, i); s_real = 2.0 * sin(alpha_h);
    rows(end+1, :) = [q, alpha_h, s_real, s_real / Tv(i), abs(lam), double(abs(imag(lam)) > 1e-7), ...
        real(dlam_fd), real(c_swing + c_reset), real(c_swing), real(c_reset), ...
        real(c_geom), real(c_vel), real(c_push), errBA, abs(eA(1)), abs(eA(2)), cos(2 * a0), ...
        real((conj(lam) * (c_swing + c_reset)) / abs(lam))]; %#ok<SAGROW>
end
fprintf('decomposition: done (%.0f s)\n', toc);

hdr = 'q,alpha_h,s,v,lam_max,is_complex,dlam_dq_fd,dlam_dq_pert,c_swing,c_reset,c_reset_geom,c_reset_vel,c_reset_push,norm_BA_minus_J,absEigA1,absEigA2,cos2alpha_h,dabslam_dq';
fh = fopen(out, 'w'); fprintf(fh, '%s\n', hdr);
fprintf(fh, [repmat('%.10g,', 1, 17) '%.10g\n'], rows.');
fclose(fh);

% ---- checks and summary -------------------------------------------------------
real_rows = rows(rows(:, 6) == 0, :);   % column 6 is is_complex
fprintf('\n=== checks ===\n');
fprintf('max ||B*A - J_fd||_F over branch           : %.2e\n', max(rows(:, 14)));
fprintf('max |dlam(pert) - dlam(fd)| on real regime : %.2e  (|dlam| up to %.3f)\n', ...
        max(abs(real_rows(:, 7) - real_rows(:, 8))), max(abs(real_rows(:, 7))));
fprintf('max |c_geom+c_vel+c_push - c_reset|        : %.2e\n', ...
        max(abs(real_rows(:, 11) + real_rows(:, 12) + real_rows(:, 13) - real_rows(:, 10))));

fprintf('\n=== real regime (below v*): d lambda_max / dq and its attribution ===\n');
fprintf('%7s %10s %8s %10s | %10s %10s | %10s %10s %10s | %8s %8s\n', ...
        'q', 's(real)', 'lam', 'dlam/dq', 'swing', 'reset', 'geom', 'vel', 'push', '|eigA1|', 'cos2a_h');
for qq = [0.010 0.020 0.030 0.050 0.080 0.100 0.120 0.140]
    [~, i] = min(abs(real_rows(:, 1) - qq)); r = real_rows(i, :);
    fprintf('%7.3f %10.6f %8.5f %10.4f | %10.4f %10.4f | %10.4f %10.4f %10.4f | %8.4f %8.5f\n', ...
            r(1), r(3), r(5), r(8), r(9), r(10), r(11), r(12), r(13), r(15), r(17));
end

% integrate the contributions over the real regime: how much of the rise of
% lambda_max from the transition to s = 0.010 is due to each factor
% Exclude a neighbourhood of the coalescence, as Supplementary S6 states: the
% first-order expression is ill-conditioned there because l'r -> 0. The
% coalescence sits at q ~ 0.141, so the integration stops at 0.130. Without
% this cut the shares read 97.4 / 2.6 / 98.0 instead of 98.9 / 1.1 / 99.1.
Q_COALESCE = 0.141; Q_GUARD = 0.011;
rr = sortrows(real_rows(real_rows(:, 1) <= Q_COALESCE - Q_GUARD + 1e-9, :), 1);
q_lo = rr(1, 1); q_hi = rr(end, 1);
I = @(col) trapz(rr(:, 1), rr(:, col));      % integral over q (ascending)
dl_total = rr(1, 5) - rr(end, 5);            % lambda(s_lo) - lambda(s_hi) > 0
fprintf('\n=== integrated over the real regime q in [%.3f, %.3f] ===\n', q_lo, q_hi);
fprintf('lambda_max rises by %.4f (from %.4f to %.4f)\n', dl_total, rr(end, 5), rr(1, 5));
fprintf('  integral of -dlam/dq (fd)     : %.4f\n', -I(7));
fprintf('  swing/event factor            : %.4f  (%.1f %%)\n', -I(9),  100 * I(9)  / I(8));
fprintf('  reset factor                  : %.4f  (%.1f %%)\n', -I(10), 100 * I(10) / I(8));
fprintf('      collision geometry (alpha): %.4f  (%.1f %%)\n', -I(11), 100 * I(11) / I(8));
fprintf('      pre-impact velocity       : %.4f  (%.1f %%)\n', -I(12), 100 * I(12) / I(8));
fprintf('      prescribed push-off P(q)  : %.4f  (%.1f %%)\n', -I(13), 100 * I(13) / I(8));
%
%   Checks built in: B*A against the directly differentiated J; the perturbation
%   derivative against a finite difference of lambda along the branch; the three
%   reset parts against the total reset term. Writes data/jacobian_decomposition.csv.
C = 1.04; c = wk.constants();
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'data_matlab', 'jacobian_decomposition.csv');
h = 0.001;                                   % spacing in q, the branch parameter
tic;

% ---- the branch, s = 0.400 down to 0.010, storing the fixed points ----------
Q = 0.400:-h:0.010; n = numel(Q);   % the grid is in q, not in realised s
Z = zeros(3, n); Pv = zeros(1, n); Tv = zeros(1, n);
seed = [];
for i = 1:n
    g = wk.solve_gait(Q(i), c.GAM, c.KHIP, c.RTOL, c.ATOL, c.DELTA, seed);
    assert(~isempty(g), 'continuation failed at q=%.3f', Q(i));
    Z(:, i) = g.z_fp; Pv(i) = g.P; Tv(i) = g.T; seed = g.z_fp;
end
fprintf('branch: %d gaits (%.0f s)\n', n, toc);

% ---- A, B, J at every gait ----------------------------------------------------
A = cell(1, n); B = cell(1, n); Y = zeros(3, n);
for i = 1:n
    [ym, ~, ok] = wk.flow_to_event(Z(:, i), c.GAM, c.KHIP, c.RTOL, c.ATOL); assert(ok);
    Y(:, i) = ym;
    A{i} = wk.jacobian_flow(Z(:, i), c.GAM, c.KHIP, c.RTOL, c.ATOL, c.DELTA);
    [~, B{i}] = wk.reset_map(ym, Pv(i));
end
fprintf('factors: done (%.0f s)\n', toc);

% ---- decomposition at interior points ----------------------------------------
rows = [];
for i = 2:n-1
    q = Q(i); Ai = A{i}; Bi = B{i}; J = Bi * Ai;
    Jfd = wk.jacobian_3d(Z(:, i), c.GAM, c.KHIP, Pv(i), c.RTOL, c.ATOL, c.DELTA);
    errBA = norm(J - Jfd, 'fro');
    [V, D, W] = eig(J);
    lam = diag(D); [~, k] = max(abs(lam)); lam = lam(k); r = V(:, k); l = W(:, k);
    nrm = l' * r;
    % derivatives along the branch: central differences over neighbouring gaits
    % (S is descending, so i+1 is s-h and i-1 is s+h)
    dA = (A{i-1} - A{i+1}) / (2 * h);
    dB = (B{i-1} - B{i+1}) / (2 * h);
    c_swing = (l' * (Bi * dA) * r) / nrm;
    c_reset = (l' * (dB * Ai) * r) / nrm;
    % reset term split by argument: B = B(alpha, theta_dot-, P)
    ep = 1e-6;
    a0 = Y(1, i); w0 = Y(2, i); P0 = Pv(i);
    da = (Y(1, i-1) - Y(1, i+1)) / (2 * h);
    dw = (Y(2, i-1) - Y(2, i+1)) / (2 * h);
    dP = (Pv(i-1) - Pv(i+1)) / (2 * h);
    [~, Bap] = wk.reset_map([a0 + ep; w0; 0], P0); [~, Bam] = wk.reset_map([a0 - ep; w0; 0], P0);
    [~, Bwp] = wk.reset_map([a0; w0 + ep; 0], P0); [~, Bwm] = wk.reset_map([a0; w0 - ep; 0], P0);
    [~, BPp] = wk.reset_map([a0; w0; 0], P0 + ep); [~, BPm] = wk.reset_map([a0; w0; 0], P0 - ep);
    dB_geom = (Bap - Bam) / (2 * ep) * da;
    dB_vel  = (Bwp - Bwm) / (2 * ep) * dw;
    dB_push = (BPp - BPm) / (2 * ep) * dP;
    c_geom = (l' * (dB_geom * Ai) * r) / nrm;
    c_vel  = (l' * (dB_vel  * Ai) * r) / nrm;
    c_push = (l' * (dB_push * Ai) * r) / nrm;
    % finite-difference derivative of the dominant multiplier itself
    lp = wk.sorted_eigs(B{i-1} * A{i-1}); lm = wk.sorted_eigs(B{i+1} * A{i+1});
    dlam_fd = (lp(1) - lm(1)) / (2 * h);
    eA = wk.sorted_eigs(Ai);
    alpha_h = Z(1, i); s_real = 2.0 * sin(alpha_h);
    rows(end+1, :) = [q, alpha_h, s_real, s_real / Tv(i), abs(lam), double(abs(imag(lam)) > 1e-7), ...
        real(dlam_fd), real(c_swing + c_reset), real(c_swing), real(c_reset), ...
        real(c_geom), real(c_vel), real(c_push), errBA, abs(eA(1)), abs(eA(2)), cos(2 * a0), ...
        real((conj(lam) * (c_swing + c_reset)) / abs(lam))]; %#ok<SAGROW>
end
fprintf('decomposition: done (%.0f s)\n', toc);

hdr = 'q,alpha_h,s,v,lam_max,is_complex,dlam_dq_fd,dlam_dq_pert,c_swing,c_reset,c_reset_geom,c_reset_vel,c_reset_push,norm_BA_minus_J,absEigA1,absEigA2,cos2alpha_h,dabslam_dq';
fh = fopen(out, 'w'); fprintf(fh, '%s\n', hdr);
fprintf(fh, [repmat('%.10g,', 1, 17) '%.10g\n'], rows.');
fclose(fh);

% ---- checks and summary -------------------------------------------------------
real_rows = rows(rows(:, 6) == 0, :);
fprintf('\n=== checks ===\n');
fprintf('max ||B*A - J_fd||_F over branch           : %.2e\n', max(rows(:, 14)));
fprintf('max |dlam(pert) - dlam(fd)| on real regime : %.2e  (|dlam| up to %.3f)\n', ...
        max(abs(real_rows(:, 7) - real_rows(:, 8))), max(abs(real_rows(:, 7))));
fprintf('max |c_geom+c_vel+c_push - c_reset|        : %.2e\n', ...
        max(abs(real_rows(:, 11) + real_rows(:, 12) + real_rows(:, 13) - real_rows(:, 10))));

fprintf('\n=== real regime (below v*): d lambda_max / dq and its attribution ===\n');
fprintf('%7s %10s %8s %10s | %10s %10s | %10s %10s %10s | %8s %8s\n', ...
        'q', 's(real)', 'lam', 'dlam/dq', 'swing', 'reset', 'geom', 'vel', 'push', '|eigA1|', 'cos2a_h');
for qq = [0.010 0.020 0.030 0.050 0.080 0.100 0.120 0.140]
    [~, i] = min(abs(real_rows(:, 1) - qq)); r = real_rows(i, :);
    fprintf('%7.3f %10.6f %8.5f %10.4f | %10.4f %10.4f | %10.4f %10.4f %10.4f | %8.4f %8.5f\n', ...
            r(1), r(3), r(5), r(8), r(9), r(10), r(11), r(12), r(13), r(15), r(17));
end

% integrate the contributions over the real regime: how much of the rise of
% lambda_max from the transition to s = 0.010 is due to each factor
% Exclude a neighbourhood of the coalescence, as Supplementary S6 states: the
% first-order expression is ill-conditioned there because l'r -> 0. The
% coalescence sits at q ~ 0.141, so the integration stops at 0.130. Without
% this cut the shares read 97.4 / 2.6 / 98.0 instead of 98.9 / 1.1 / 99.1.
Q_COALESCE = 0.141; Q_GUARD = 0.011;
rr = sortrows(real_rows(real_rows(:, 1) <= Q_COALESCE - Q_GUARD + 1e-9, :), 1);
q_lo = rr(1, 1); q_hi = rr(end, 1);
I = @(col) trapz(rr(:, 1), rr(:, col));      % integral over q (ascending)
dl_total = rr(1, 5) - rr(end, 5);            % lambda(s_lo) - lambda(s_hi) > 0
fprintf('\n=== integrated over the real regime q in [%.3f, %.3f] ===\n', q_lo, q_hi);
fprintf('lambda_max rises by %.4f (from %.4f to %.4f)\n', dl_total, rr(end, 5), rr(1, 5));
fprintf('  integral of -dlam/dq (fd)     : %.4f\n', -I(7));
fprintf('  swing/event factor            : %.4f  (%.1f %%)\n', -I(9),  100 * I(9)  / I(8));
fprintf('  reset factor                  : %.4f  (%.1f %%)\n', -I(10), 100 * I(10) / I(8));
fprintf('      collision geometry (alpha): %.4f  (%.1f %%)\n', -I(11), 100 * I(11) / I(8));
fprintf('      pre-impact velocity       : %.4f  (%.1f %%)\n', -I(12), 100 * I(12) / I(8));
fprintf('      prescribed push-off P(q)  : %.4f  (%.1f %%)\n', -I(13), 100 * I(13) / I(8));
fprintf('\nsaved: %s  (%.0f s total)\n', out, toc);
