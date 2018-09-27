clear all;
close all;
clc;

% Run 1: x = -2,lrate = 0.1,threshold = 10^-2
[x,cost] = runGradientDescent(-2,0.1,0.01);

% Analysis

figure
hold on;
plot(cost);
title('Cost vs Iterations elapsed for different eta');
xlabel('Iterations');
ylabel('Cost');
ylim([0 10])
xlim([1 20])

% Run 2: x = -2,lrate = 1,threshold = 10^-2
[x,cost] = runGradientDescent(-2,1,0.01);
plot(cost);
text(15,4.2,'x oscillates b/w -2 and 2');

% Run 3: x = -2,lrate = -1,threshold = 10^-2
[x,cost] = runGradientDescent(-2,1.2,0.01);
plot(cost);

legend('convergence','osciallation','divergence');

% Gradient Descent (fixed gradient - 2x)
function[x,cost] = runGradientDescent(xinit,lrate,update_threshold)
max_iters = 50;
iter = 1;
x = xinit;
cost = [x * x];

gradient = 2 * x;
update = lrate * gradient;

while(iter < max_iters && (abs(update) > update_threshold))
    iter = iter + 1;
    x = x - update;
    cost = [cost x * x];
    gradient = 2 * x;
    update = lrate * gradient;
end
end
