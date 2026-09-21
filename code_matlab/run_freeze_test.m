%RUN_FREEZE_TEST  Which factor carries the step-length dependence of lambda_max?
%
%   J(s) = B(s) A(s). Freeze one factor at a reference step length and let the
%   other vary along the branch:
%       lambda_A-frozen(s) = max eig( B(s)  A(s_ref) )   swing/event map held fixed
%       lambda_B-frozen(s) = max eig( B(s_ref) A(s) )    reset map held fixed
%   If the first tracks the true lambda_max(s) and the second does not, the
%   s-dependence enters through the reset (collision geometry and push-off); the
%   swing/event map then sets the level (the coefficient) but not the trend.
%   Writes data/jacobian_freeze_test.csv.
c = wk.constants();
here = fileparts(mfilename('fullpath'));
out = fullfile(here, '..', 'data_matlab', 'jacobian_freeze_test.csv');
Q = 0.140:-0.001:0.010; n = numel(Q);   % grid in q, the branch parameter
tic;
% branch by continuation from s = 0.40
seed = []; g0 = wk.solve_gait(0.40); seed = g0.z_fp;
for s = 0.399:-0.001:0.141
    g = wk.solve_gait(s, [], [], [], [], [], seed); assert(~isempty(g)); seed = g.z_fp;
end
A = cell(1, n); B = cell(1, n); lam = zeros(1, n); v = zeros(1, n);
AH = zeros(1, n); SR = zeros(1, n);
for i = 1:n
    g = wk.solve_gait(Q(i), [], [], [], [], [], seed); assert(~isempty(g)); seed = g.z_fp;
    AH(i) = g.alpha_h; SR(i) = g.s;
    [ym, ~, ok] = wk.flow_to_event(g.z_fp); assert(ok);
    A{i} = wk.jacobian_flow(g.z_fp);
    [~, B{i}] = wk.reset_map(ym, g.P);
    e = wk.sorted_eigs(B{i} * A{i}); lam(i) = abs(e(1)); v(i) = g.v;
end
fprintf('branch and factors: %.0f s\n', toc);

REFS = [0.010, 0.050, 0.100];
fh = fopen(out, 'w');
fprintf(fh, 'q,alpha_h,s,v,lam_true,lam_Afrozen_q010,lam_Bfrozen_q010,lam_Afrozen_q050,lam_Bfrozen_q050,lam_Afrozen_q100,lam_Bfrozen_q100\n');
LA = zeros(numel(REFS), n); LB = zeros(numel(REFS), n);
for k = 1:numel(REFS)
    [~, ir] = min(abs(Q - REFS(k)));
    for i = 1:n
        ea = wk.sorted_eigs(B{i} * A{ir}); LA(k, i) = abs(ea(1));
        eb = wk.sorted_eigs(B{ir} * A{i}); LB(k, i) = abs(eb(1));
    end
end
for i = 1:n
    fprintf(fh, '%.3f,%.10g,%.10g,%.10g,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,%.8f\n', ...
            Q(i), AH(i), SR(i), v(i), lam(i), ...
            LA(1, i), LB(1, i), LA(2, i), LB(2, i), LA(3, i), LB(3, i));
end
fclose(fh);

fprintf('\n=== 1 - lambda_max along the real regime: true vs one factor frozen at s_ref ===\n');
fprintf('%7s %10s | %10s %10s | %10s %10s | %10s %10s\n', 's', '1-lam', 'A@0.010', 'B@0.010', 'A@0.050', 'B@0.050', 'A@0.100', 'B@0.100');
for s = [0.010 0.020 0.030 0.050 0.080 0.100 0.120 0.140]
    [~, i] = min(abs(Q - s));
    fprintf('%7.3f %10.5f | %10.5f %10.5f | %10.5f %10.5f | %10.5f %10.5f\n', Q(i), 1 - lam(i), ...
            1 - LA(1, i), 1 - LB(1, i), 1 - LA(2, i), 1 - LB(2, i), 1 - LA(3, i), 1 - LB(3, i));
end
fprintf('\n(A@x: swing/event map frozen at s=x, reset varies;  B@x: reset frozen at s=x, swing varies)\n');
fprintf('saved: %s  (%.0f s)\n', out, toc);
