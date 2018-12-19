function y = scale(x, t)
% x: data tobe scaled, t: time seriers

y = x * 1e14;
return

%% naive scale, zscore method
y = (x - mean(x(:))) / std(x(:));
return

%% scale data using mean and std value of t<0
%% data size is 12x1001, 12 samples, 1001 ms
x_ = x(:, t<0); % cut data of t<0
m_ = mean(x_, 2); % cal mean
s_ = std(x_, [], 2); % cal std
% scale
y = (x - repmat(m_, 1, size(x, 2)) ./ repmat(s_, 1, size(x, 2)));

end