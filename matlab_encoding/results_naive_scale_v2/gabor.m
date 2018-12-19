function y = gabor(t, para)
% t: time seriers, para: parameters
%% initial parameters
A = para(1);
A0 = para(2);
t0 = para(3);
d = para(4);
w = para(5);
p = para(6);
w0 = para(7);
p0 = para(8);

%% calculate parts
E = exp(-((t-t0).^2) / d^2);
C = cos(w*t + p);
C0 = cos(w0*t + p0);

%% calculate Gabor from parts
% y = A * E .* C + A0;
y = A * E .* C + A0 * C0;

%% make output column
if isrow(y)
    y = y';
end

end

