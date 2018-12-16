function out = cal_loss(diff)
out = diag((diff' * diff) .^ 0.5);
end