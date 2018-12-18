function loss = norm2loss(para_arr)
% para_arr: parameters as array
%% load data
% t: time seriers, d: data
% from harddriver, since called by fmincon 
load tmp_data t d

%% calculate Gabor function
y = gabor(t, para_arr);

%% calcualte diff between estimation and data
diff = bsxfun(@plus, d, -y);

%% cal culate loss as Eucidean distance
loss = sum(diag((diff' * diff) .^ 0.1));

end

