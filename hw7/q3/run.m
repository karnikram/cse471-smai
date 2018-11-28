clear all;
close all;
clc;

% Generating data samples
x = rand(2,10);
y = 10 .* rand(10,1);

figure
xlim([0 1])
ylim([0 1])
zlim([0 10])
plot3(x(1,:),x(2,:),y,'o');
xlabel('x1')
ylabel('x2')
zlabel('y')
title('Data points')

% Converting to homogeneous form xw = 0
w = [2 -1 2]';
x(3,:) = 1;
x = x'; 

% Gradient descent
l_rate = 1;
n = size(x,1);
MAX_ITERS = 500;
n_iters = 0;
disp('Initial weight');
disp(w);
costs = [calcCost(x,w,y)];
disp('Initial cost');
disp(costs);
update = l_rate * calcJacobian(x,w,y);

while(n_iters < MAX_ITERS && norm(update) > 0.001)
    w = w - update;
    costs = [costs calcCost(x,w,y)];
    n_iters = n_iters + 1;
    update = l_rate * calcJacobian(x,w,y);
end

disp('Gradient Descent Result:')
disp('Weight estimated:')
disp(w)
disp('Number of iterations elapsed')
disp(n_iters)
disp('Final cost')
disp(costs(end))

figure
plot(costs);

% Newton's Method
w = [2 -1 2]';
disp('Initial weight');
disp(w);
n_iters = 0;
MAX_ITERS = 500;
costs = calcCost(x,w,y);
disp('Initial cost');
disp(costs);
update = inv(calcHessian(x,w,y)) * calcJacobian(x,w,y);
while(n_iters < MAX_ITERS && norm(update) > 0.001)
    w = w - update;
    costs = [costs calcCost(x,w,y)];
    n_iters = n_iters + 1;
    update = inv(calcHessian(x,w,y)) * calcJacobian(x,w,y);
end
w_est = w;
disp('Newton Method Result:')
disp('Weight estimated:')
disp(w_est)
disp('Number of iterations elapsed')
disp(n_iters)
disp('Final cost')
disp(costs(end))
hold on;

plot(costs);
xlabel('Iterations')
ylabel('Cost')
title('Convergence curves')
legend('Gradient Descent','Newtons Method')

% Plot hyperplane
figure
xlim([0 1])
ylim([0 1])
zlim([0 10])
plot3(x(:,1),x(:,2),y,'o');
xlabel('x1')
ylabel('x2')
zlabel('y')

[X,Y] = meshgrid(0:0.01:1,0:0.01:1);

Z = X*w(1) + Y*w(2) + w(3);
hold on;
surf(X,Y,Z);
title('Estimated Hyperplane');

% Error curves
% cost vs w1
j = 0.1;
w = w_est;
w1 = -20;
W1 = [w1];
costs = [];
W1 = [];

for i = 1 : 400
    w1 = w1 + j;
    W1 = [W1 w1];
    w(1) = w1;
    costs = [costs calcCost(x,w,y)];
end
figure
subplot(3,1,1)
plot(W1,costs)
xlabel('w1')
ylabel('cost')
title('cost vs w1');

% cost vs w2
w = w_est;
costs = [];
j = 0.1;
w2 = -20;
W2 = [];
for i = 1 : 400
    w2 = w2 + j;
    W2 = [W2 w2];
    w(2) = w2;
    costs = [costs ;calcCost(x,w,y)];
end

subplot(3,1,2)
plot(W2,costs)
xlabel('w2')
ylabel('cost')
title('cost vs w2');

% cost vs w3
w = w_est;
costs  = [];
j = 0.1;
w3 = -20;
W3 = [];
for i = 1 : 400
    w3 = w3 + j;
    W3 = [W3 w3];
    w(3) = w3;
    costs = [costs calcCost(x,w,y)];
end
subplot(3,1,3)
plot(W3,costs)
xlabel('w3')
ylabel('cost')
title('cost vs w3')

% Error surface
w1 = (w_est(1) - 20):0.1:(w_est(1) + 20);
w2 = (w_est(2) - 20):0.1:(w_est(2) + 20);
w3 = w_est(3);
costs = zeros(size(w2,1),size(w2,1));

% w3 is kept constant
for i = 1 : size(w2,2)
    for j = 1 : size(w1,2)
        w = [w1(j) w2(i) w3]';
        costs(i,j) = calcCost(x,w,y);
    end
end

[W1,W2] = meshgrid(w1,w2);
figure
c = W1.*W2;
surf(W1,W2,costs,'Edgecolor','None','Facecolor',[0 0 1]);
xlabel('w1')
ylabel('w2')
zlabel('cost')
title('Error surface with w3 kept constant')

% Cost function
function [cost] = calcCost(x,w,y)
cost = (y - x * w)' * (y - x * w);
cost = 1 / 2 * size(x,1) * cost;
end

% Jacobian
function [j] = calcJacobian(x,w,y)
j = -1 * 1/size(x,1) * x' * (y - x * w);
end

% Hessian
function [h] = calcHessian(x,w,y)
h = 1/size(x,1) * x' * x;
end
