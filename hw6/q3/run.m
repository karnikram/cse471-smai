close all;

clear;

clc;



%% Bayesian optimal decision boundary



% mean and covariance

rng(2018701003)

mean_n = [1.2; 1.2];

mean_p = [1.2; 1.2];

covar_p = [2 -1; -1 37.75];

covar_n = [4 -2.2; -2.2 40];

% mean_n = [-2.2; -2.2];

% mean_p = [1.2; 1.2];

% covar_p = [2 0.1; 0.1 2];

% covar_n = [8 1.5; 1.5 25];



% prior probabilities of positve and neagative classes

prior_p = 0.5;

prior_n = 1 - prior_p;



% 2-D synthetic data generation

data_p = mvnrnd(mean_p,covar_p,500);

data_n = mvnrnd(mean_n,covar_n,500);



inv_covar_p = inv(covar_p);

det_covar_p = det(covar_p);

term_quad_p = (-1/2)*(inv_covar_p);

term_lin_p = inv_covar_p*mean_p;

term_const_p = (-1/2)*(mean_p'*inv_covar_p*mean_p) -(1/2)*log(det_covar_p) + log(prior_p);



inv_covar_n = inv(covar_n);

det_covar_n = det(covar_n);

term_quad_n = (-1/2)*(inv_covar_n);

term_lin_n = inv_covar_n*mean_n;

term_const_n = (-1/2)*(mean_n'*inv_covar_n*mean_n) -(1/2)*log(det_covar_n) + log(prior_n);



syms x1 x2;

x = [x1;x2];

dec_boundary = x'*term_quad_p*x + term_lin_p'*x + term_const_p == x'*term_quad_n*x + term_lin_n'*x + term_const_n;



% % Linear decision boundary weights if equal covariance

% mid_var = (mean_p - mean_n);

% inv_var = inv(covar_p); 

% weight = inv_var*mid_var;

% weight_b = -weight'*((mean_p + mean_n)/2 - (log(prior_p/prior_n)*mid_var)/(mid_var'*inv_var*mid_var));

% weight = [weight; weight_b];

% x_1 = -5:0.1:5;

% x_2 = -(weight(1)*x_1 + weight(3))/(weight(2));



x_2 = -25:1:25;

x_1 = zeros(size(x_2,2),2);

for i = 1:size(x_2,2)

    m = subs(dec_boundary,x2,x_2(1,i));

    x_1(i,:) = double(solve(m,x1))';

end



%% plot of samples and decision boundary



figure(1);

hold on;

xlim([-8 8]);

ylim([-25 25]);

plot(data_p(1:200,1),data_p(1:200,2),'bo','LineWidth',2);

plot(data_n(1:200,1),data_n(1:200,2),'ro','LineWidth',2);

plot(x_1(:,1),x_2,'g');

plot(x_1(:,2),x_2,'g');

xlabel('x1');

ylabel('x2');

legend('Positive samples','Negative samples','Bayesian decision boundary')
