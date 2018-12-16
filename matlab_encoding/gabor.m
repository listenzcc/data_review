function y = gabor(t, para)
% t: time seriers, para: parameters
%% initial parameters
A = para(1);
A0 = para(2);
t0 = para(3);
d = para(4);
w = para(5);
p = para(6);

%% calculate parts
E = exp(-((t-t0).^2) / d^2);
C = cos(w*t + p);

%% calculate Gabor from parts
y = A * E .* C + A0;

%% make output column
if isrow(y)
    y = y';
end

end

