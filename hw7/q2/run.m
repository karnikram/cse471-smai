clear all;
clc;
close all;

% Features
x1 = [1 1 1];
x2 = [-1 -1 1];
x3 = [2 2 1];
x4 = [-2 -2 1];
x5 = [-1 1 1];
x6 = [1 -1 1];

x = [x1' x2' x3' x4' x5' x6'];

% Labels
y = [-1 -1 1 -1 1 1];

% Initialization
weight = [1 0 -1]';
l_rate = 0.5;
MAX_ITERS = 100;
num_iters = 0;
costs = [];

misclassified_indices = testModel(x,weight,y);

% Gradient Descent
while(size(misclassified_indices,2) ~= 0 && num_iters < MAX_ITERS)
    num_iters = num_iters + 1;
    sum = 0;
    for j = 1 : size(misclassified_indices,2)
        i = misclassified_indices(1,j);
        sum = sum + y(i) * x(:,i);
    end
    weight = weight + l_rate * sum;
    misclassified_indices = testModel(x,weight,y);
    cost = calcCost(x,weight,y,misclassified_indices);
    costs = [costs cost];
end

figure
plot(costs);
title('Oscillating cost curve')
xlabel('Iterations')
ylabel('Cost')

disp('Costs at different iterations: ')
costs

% Cost function
function [cost] = calcCost(x,weight,y,indices)
cost = 0;
    for j = 1 : size(indices,2)
         i = indices(1,j);
        cost = cost + y(i) * weight' * x(:,i) * -1;
    end
end

% Test for misclassified samples
function [misclassified_indices] = testModel(x,weight,y)
misclassified_indices=[];
for i = 1:6
    out = getOutput(x(:,i),weight);
    if(out ~= y(i))
        misclassified_indices = [misclassified_indices i];
    end
end
end

% Get class label from classifier
function [output] = getOutput(x,weight)
    output = weight' * x;
    if(output >= 0)
        output = 1;
    else
        output = -1;
    end
end
