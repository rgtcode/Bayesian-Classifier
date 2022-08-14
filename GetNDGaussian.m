function y = GetNDGaussian(x, m, C)

ddim = length(x);
k = ddim;

den = (2*pi)^k * det(C);
den = sqrt(den);

exponent = 0.5 * (x-m)'* inv(C) * (x-m);

y = exp(-exponent)/den;

end