clear all;
close all;
clc;

% Run 1: x = -2,lrate = 0.1,threshold = 10^-1
[x,cost,time] = runGradientDescent(-2,0.1,0.1);

% Analysis

figure
plot(time,cost);
title({'Cost vs Time elapsed';'x_i = -2,eta = 0.1'});
xlabel('Time (seconds)');
ylabel('Cost');

v_steps = size(cost,2);

% Run 2: x = -2,lrate = 0.1,threshold = 10^-2
[x,cost,time] = runGradientDescent(-2,0.1,0.01);
v_steps = [v_steps size(cost,2)];

% Run 3: x = -2,lrate = 0.1,threshold = 10^-3
[x,cost,time] = runGradientDescent(-2,0.1,0.001);
v_steps = [v_steps size(cost,2)];

% Run 3: x = -2,lrate = 0.1,threshold = 10^-4
[x,cost,time] = runGradientDescent(-2,0.1,0.0001);
v_steps = [v_steps size(cost,2)];

criteria = [0.1, 0.01, 0.001, 0.0001];
figure
plot(v_steps, criteria,'-*');
set(gca, 'YScale', 'log')
xlabel('No. of iterations');
ylabel('$\vert \eta\nabla J \vert$ threshold','interpreter','latex');
title({'Convergence criteria vs Iterations','x_i = -2,eta = 0.1'});

% Gradient Descent (fixed gradient - 2x)
function[x,cost,time] = runGradientDescent(xinit,lrate,update_threshold)
max_iters = 100;
iter = 0;
x = xinit;
cost = [];
time = [];

tic
gradient = 2 * x;
update = lrate * gradient;

while(iter < max_iters && (abs(update) > update_threshold))
    iter = iter + 1;
    cost = [cost x * x];
    x = x - update;
    time = [time toc];
    gradient = 2 * x;
    update = lrate * gradient;
end
end
