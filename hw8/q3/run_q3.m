clear all;
close all;
clc;

mu1 = [4;2];
mu2 = [2;2];
cov = [1 -2; -2 10];
pw1 = 0.5;
pw2 = 0.5;

R1 = mvnrnd(mu1,cov,500);
R2 = mvnrnd(mu2,cov,500);
X = [R1;R2];

% Bayesian decision surface
w = inv(cov) * (mu1 - mu2);
x0 = 1/2*(mu1 + mu2)- log(pw1/pw2)./((mu1 - mu2)'*inv(cov)*(mu1 - mu2)) * (mu1 - mu2);
w0 = -w' * x0; % Taking x = [0;0]
 
W = [w;w0];
 
x1 = -5:0.05:5;
x2 = -(W(1) * x1 + W(3))/W(2);
 
% Accuracy of Bayes
bayes_miss = 0;
for i = 1:size(R1,1)
    if(W' * ([R1(i,:)';1] - [x0;1]) < 0)
    bayes_miss = bayes_miss + 1;
    end
end

for i = 1:size(R2,1)
    if(W' * ([R2(i,:)';1] - [x0;1]) > 0)
    bayes_miss = bayes_miss + 1;
    end
end

bayes_accuracy = (((1000 - bayes_miss)/1000)*100);
W_bayes = W;

figure
title('Train samples');
xlim([-10 10]);
ylim([-10 10]);
hold on;
plot(R1(:,1),R1(:,2),'+')
plot(R2(:,1),R2(:,2),'o')
plot(x1,x2);
xlabel('x1')
ylabel('x2')

% Logistic regression with bias
W = [1;1;1];
Y = [ones(500,1);zeros(500,1)];
X = [X ones(size(Y,1),1)];
lrate = 0.0001;
threshold = 0.001;
iter = 0;
MAX_ITERS = 1000;

update = -lrate * calcGradient(X,W,Y);

while(iter < MAX_ITERS && norm(update) > threshold)
    W = W + update;
    iter = iter + 1;
    update = -lrate * calcGradient(X,W,Y);
end

W_lr1 = W;

% Accuracy of logistic regression with bias
lr1_miss = 0;
for i = 1:500
    if(W_lr1' * X(i,:)' < 0)
    lr1_miss = lr1_miss + 1;
    end
end

for i = 501:1000
    if(W_lr1' * X(i,:)' > 0)
    lr1_miss = lr1_miss + 1;
    end
end

lr1_accuracy = (((1000 - lr1_miss)/1000)*100);

x1 = -5:0.05:5;
x2 = -(W_lr1(1) * x1 + W_lr1(3))/W_lr1(2);
plot(x1,x2);

% Logistic regression without bias
W = [1;1];
Y = [ones(500,1);zeros(500,1)];
X = X(:,1:2);
lrate = 0.0001;
threshold = 0.001;
iter = 0;
MAX_ITERS = 1000;

update = -lrate * calcGradient(X,W,Y);

while(iter < MAX_ITERS && norm(update) > threshold)
    W = W + update;
    iter = iter + 1;
    update = -lrate * calcGradient(X,W,Y);
end

W_lr2 = W;

% Accuracy of logistic regression without bias
lr2_miss = 0;
for i = 1:500
    if(W_lr2' * X(i,:)' < 0)
    lr2_miss = lr2_miss + 1;
    end
end

for i = 501:1000
    if(W_lr2' * X(i,:)' > 0)
    lr2_miss = lr2_miss + 1;
    end
end

lr2_accuracy = (((1000 - lr2_miss)/1000)*100);

x1 = -5:0.05:5;
x2 = -(W_lr2(1) * x1)/W_lr2(2);
plot(x1,x2);

legend('omega1','omega2','Bayesian decision boundary','Logistic regression with bias', 'Logistic regression without bias');

disp('Performance on training samples (accuracy percentage)');
disp('Bayes classifier:');
disp(bayes_accuracy);
disp('Logistic regression with bias');
disp(lr1_accuracy);
disp('Logistic regression without bias');
disp(lr2_accuracy);

% Performance on test samples

mu1 = [4;2];
mu2 = [2;2];
cov = [1 -2; -2 10];
pw1 = 0.5;
pw2 = 0.5;

R1 = mvnrnd(mu1,cov,500);
R2 = mvnrnd(mu2,cov,500);
X = [R1;R2];

% Accuracy of Bayes
x0 = 1/2*(mu1 + mu2)- log(pw1/pw2)./((mu1 - mu2)'*inv(cov)*(mu1 - mu2)) * (mu1 - mu2);

x1 = -5:0.05:5;
x2 = -(W_bayes(1) * x1 + W_bayes(3))/W_bayes(2);

bayes_miss = 0;
for i = 1:size(R1,1)
    if(W_bayes' * ([R1(i,:)';1] - [x0;1]) < 0)
    bayes_miss = bayes_miss + 1;
    end
end

for i = 1:size(R2,1)
    if(W_bayes' * ([R2(i,:)';1] - [x0;1]) > 0)
    bayes_miss = bayes_miss + 1;
    end
end

bayes_accuracy_test = (((1000 - bayes_miss)/1000)*100);

figure
title('Test samples')
xlim([-10 10]);
ylim([-10 10]);
hold on;
plot(R1(:,1),R1(:,2),'+')
plot(R2(:,1),R2(:,2),'o')
plot(x1,x2);
xlabel('x1')
ylabel('x2')

% Accuracy of logistic regression with bias
Y = [ones(500,1);zeros(500,1)];
X = [X ones(size(Y,1),1)];

lr1_miss = 0;
for i = 1:500
    if(W_lr1' * X(i,:)' < 0)
    lr1_miss = lr1_miss + 1;
    end
end

for i = 501:1000
    if(W_lr1' * X(i,:)' > 0)
    lr1_miss = lr1_miss + 1;
    end
end

lr1_accuracy_test = (((1000 - lr1_miss)/1000)*100);

x1 = -5:0.05:5;
x2 = -(W_lr1(1) * x1 + W_lr1(3))/W_lr1(2);
plot(x1,x2);

% Accuracy of logistic regression without bias
Y = [ones(500,1);zeros(500,1)];
X = X(:,1:2);

lr2_miss = 0;
for i = 1:500
    if(W_lr2' * X(i,:)' < 0)
    lr2_miss = lr2_miss + 1;
    end
end

for i = 501:1000
    if(W_lr2' * X(i,:)' > 0)
    lr2_miss = lr2_miss + 1;
    end
end

lr2_accuracy_test = (((1000 - lr2_miss)/1000)*100);

x1 = -5:0.05:5;
x2 = -(W_lr2(1) * x1)/W_lr2(2);
plot(x1,x2);

legend('omega1','omega2','Bayesian decision boundary','Logistic regression with bias', 'Logistic regression without bias');

disp('Performance on test samples (accuracy percentage)');
disp('Bayes classifier:');
disp(bayes_accuracy_test);
disp('Logistic regression with bias');
disp(lr1_accuracy_test);
disp('Logistic regression without bias');
disp(lr2_accuracy_test);

figure
c = categorical({'Bayes','LR with bias','LR without bias'});
bar(c,[bayes_accuracy,bayes_accuracy_test;lr1_accuracy,lr1_accuracy_test;lr2_accuracy,lr2_accuracy_test]);
legend('Training samples','Test samples');

function [dJ] = calcGradient(X,W,Y)
    H = zeros(size(Y,1),1);
    for i = 1:size(H,1)
        H(i) = 1/(1 + exp(-W'*X(i,:)'));
    end
    
    dJ = X'*(H - Y);
end