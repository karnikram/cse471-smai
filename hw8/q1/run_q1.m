clear all;
close all;
clc;

x = 2 * rand([1 1000]) - 1;
X = [x' ones(1000,1)];
Y = ones(1000,1);
for i = 1:1000
    if(x(i) < 0)
        Y(i) = 0;
    end
end

[W1,W2] = meshgrid(-5:0.1:5);
J = zeros(size(W1));

% Linear Regression
for i = 1 : size(W1,1)
    for j = 1 : size(W1,2)
        W = [W1(i,j) W2(i,j)]';
        J(i,j) = (Y - X*W)' * (Y - X*W);
    end
end
figure
subplot(1,2,1);
surf(W1,W2,J,'EdgeColor','None');
title('Linear Regression Cost');
xlabel('w1');
ylabel('w2');
zlabel('J');

% Logistic Regression (not really)
for i = 1 : size(W1,1)
    for j = 1 : size(W1,2)
        W = [W1(i,j) W2(i,j)]';
        G = calcG(X,W);
        J(i,j) = (Y - G)' * (Y - G);
    end
end

subplot(1,2,2);
surf(W1,W2,J,'EdgeColor','None');
title('(y - g(w^Tx))^2 Cost');
xlabel('w1');
ylabel('w2');
zlabel('J');

function [G] = calcG(X,W)
    G = ones(size(X,1),1);
    for i = 1:numel(G)
        G(i) = 1/(1 + exp(-1 * W' * X(i,:)'));
    end
end