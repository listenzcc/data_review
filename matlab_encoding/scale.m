function y = scale(x)
y = (x - mean(x(:))) / std(x(:));
end