function loss = norm2loss(para_arr)
load tmp_data t d

y = gabor(t, para_arr);

diff = bsxfun(@plus, d, -y);

loss = sum(diag((diff' * diff) .^ 0.5));

end

