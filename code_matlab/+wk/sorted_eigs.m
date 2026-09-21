function lam = sorted_eigs(J)
%SORTED_EIGS  Eigenvalues of J sorted by descending magnitude.
%   Ties (a complex-conjugate pair) are broken with the positive imaginary
%   part first, matching the LAPACK ordering the reference implementation relies on.
lam = eig(J);
[~, idx] = sortrows([-abs(lam), -imag(lam)]);
lam = lam(idx);
end
