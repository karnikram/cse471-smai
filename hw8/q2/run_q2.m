clear all;
clc;
close all;

mu1 = [1;1];
mu2 = [-1;1];
cov = [1 -2; -2 10];
pw1 = 0.5;
pw2 = 0.5;

R1 = mvnrnd(mu1,cov,500);
R2 = mvnrnd(mu2,cov,500);
R = [R1;R2];

% Bayesian decision surface
w = inv(cov) * (mu1 - mu2);
x0 = 1/2*(mu1 + mu2)- log(pw1/pw2)./((mu1 - mu2)'*inv(cov)*(mu1 - mu2)) * (mu1 - mu2);
w0 = -w' * x0; % Taking x = [0;0]
 
W = [w;w0];
 
x1 = -5:0.05:5;
x2 = -(W(1) * x1 + W(3))/W(2);
 
% Accuracy
miss = 0;
for i = 1:size(R1,1)
    if(W' * ([R1(i,:)';1] - [x0;1]) < 0)
    miss = miss + 1;
    end
end

for i = 1:size(R2,1)
    if(W' * ([R2(i,:)';1] - [x0;1]) > 0)
    miss = miss + 1;
    end
end

disp('Number of misclassifications')
disp(miss);
disp('Accuracy percentage (on training set)');
disp(((1000 - miss)/1000)*100);

figure
hold on;
xlim([-10 10]);
ylim([-10 10]);
plot(R1(:,1),R1(:,2),'+')
plot(R2(:,1),R2(:,2),'o')
plot(x1,x2);
xlabel('x1')
ylabel('x2')
legend('omega1','omega2','Bayesian decision boundary')
