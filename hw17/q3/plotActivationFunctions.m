clear variables
close all
clc

X = -100:99;

%% Sigmoid

phi = 1 ./ (1 + exp(-1 .* X));
dphi = phi .* ( 1 - phi);

figure
subplot(3,1,1)
plot(X,phi)
hold on
plot(X,dphi)
xlabel('$x$','Interpreter','latex')
ylabel('$\phi$','Interpreter','latex')
title('Sigmoid')
legend('Function','Derivative')

%% Hyperbolic tangent

phi = (exp(X) - exp(-1.*X)) ./ (exp(X) + exp(-1 .* X));
dphi = 1 - phi .* phi;

subplot(3,1,2)
plot(X,phi)
hold on
plot(X,dphi)
xlabel('$x$','Interpreter','latex')
ylabel('$\phi$','Interpreter','latex')
title('Hyperbolic tangent')
legend('Function','Derivative')

%% ReLU

phi = max(X,0);
dphi = [zeros(1,100) ones(1,100)];

subplot(3,1,3)
plot(X,phi)
hold on
plot(X,dphi)
xlabel('$x$','Interpreter','latex')
ylabel('$\phi$','Interpreter','latex')
title('ReLU')
legend('Function','Derivative')
