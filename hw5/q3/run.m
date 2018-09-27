clear all
close all
clc

% Non - convex function : y^2 - x^2
[X,Y] = meshgrid(-1:0.01:1);
Z = Y.^2 - X.^2;

figure
hold on
xlim([-1 1]);
ylim([-1 1]);
zlim([-1 1]);
surf(X,Y,Z,'EdgeColor','None');
title('$z = y^2 - x^2$','Interpreter','latex');
xlabel('$x$','Interpreter','latex');
ylabel('$y$','Interpreter','latex');
zlabel('$z$','Interpreter','latex');

% Run 1
[v_x, v_cost] = runGradientDescent([0;-0.3]);
plot3(v_x(1,:),v_x(2,:),v_x(2,:).^2 - v_x(1,:).^2,'g');
p1 = plot3(v_x(1,1),v_x(2,1),v_x(2,1)^2 - v_x(1,1)^2,'.b');
p2 = plot3(v_x(1,end),v_x(2,end), v_x(2,end)^2 - v_x(1,end)^2, '.r');
disp('Final cost after run 1:');
disp(v_cost(end));

% Run 2
[v_x, v_cost] = runGradientDescent([-0.2;-0.3]);
plot3(v_x(1,:),v_x(2,:),v_x(2,:).^2 - v_x(1,:).^2,'g');
plot3(v_x(1,1),v_x(2,1),v_x(2,1)^2 - v_x(1,1)^2,'.b');
plot3(v_x(1,end),v_x(2,end), v_x(2,end)^2 - v_x(1,end)^2, '.r');
disp('Final cost after run 2:');
disp(v_cost(end));

% Run 3
[v_x, v_cost] = runGradientDescent([0.2;-0.3]);
plot3(v_x(1,:),v_x(2,:),v_x(2,:).^2 - v_x(1,:).^2,'g');
plot3(v_x(1,1),v_x(2,1),v_x(2,1)^2 - v_x(1,1)^2,'.b');
plot3(v_x(1,end),v_x(2,end), v_x(2,end)^2 - v_x(1,end)^2, '.r');
disp('Final cost after run 3:');
disp(v_cost(end));

% Run 4
[v_x, v_cost] = runGradientDescent([-0.05;0.4]);
plot3(v_x(1,:),v_x(2,:),v_x(2,:).^2 - v_x(1,:).^2,'g');
plot3(v_x(1,1),v_x(2,1),v_x(2,1)^2 - v_x(1,1)^2,'.b');
plot3(v_x(1,end),v_x(2,end), v_x(2,end)^2 - v_x(1,end)^2, '.r');
disp('Final cost after run 4:');
disp(v_cost(end));

% Run 5
[v_x, v_cost] = runGradientDescent([0;0.5]);
plot3(v_x(1,:),v_x(2,:),v_x(2,:).^2 - v_x(1,:).^2,'g');
plot3(v_x(1,1),v_x(2,1),v_x(2,1)^2 - v_x(1,1)^2,'.b');
plot3(v_x(1,end),v_x(2,end), v_x(2,end)^2 - v_x(1,end)^2, '.r');
disp('Final cost after run 5:');
disp(v_cost(end));

legend([p1 p2], {'Initial','Final'});

% Gradient descent (fixed gradient of [-2x;2y])
function [v_x,v_cost] = runGradientDescent(xinit)
lrate = 0.01;
max_iters = 100;
iter = 0;
x = xinit;
v_x = x;
update_threshold = 0.00001;

gradient = [- 2 * x(1); 2 * x(2)];
update = lrate * gradient;
v_cost = [];

while(iter < max_iters && (norm(update) > update_threshold))
    iter = iter + 1;
    v_cost = [v_cost (x(2)^2 - x(1)^2)];
    x = x - update;
    v_x = [v_x x];
    gradient = [- 2 * x(1); 2 * x(2)];
    update = lrate * gradient;
end
end