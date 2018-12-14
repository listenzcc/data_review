close all
clear
clc

t = linspace(-0.2, 0.8, 10001);

para = struct;
para.A = 1;
para.A0 = 2;
para.t0 = 0.2;
para.d = 0.3;
para.w = 10;
para.p = 0.3;

y = gabor(t, str2arr(para));

figure,
plot(t, y)

function para_arr = str2arr(para_str)

para_arr(1) = para_str.A;
para_arr(2) = para_str.A0;
para_arr(3) = para_str.t0;
para_arr(4) = para_str.d;
para_arr(5) = para_str.w;
para_arr(6) = para_str.p;

end
