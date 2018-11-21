clear all;
close all;
clc;

x1 = [1;2];
x2 = [2;3];
x3 = [3;4];
x4 = [-1;0];
x5 = [-2;-1];
x6 = [-3;-2];

X = [x1 x2 x3 x4 x5 x6];

% Mean normalization
mu = mean(X,2);
for i = 1:size(X,2)
    X(:,i) = X(:,i) - mu;
end

%% Principal Component Analysis

k = 1;
cov = X * X';
[U,S,V] = svd(cov);

Ured = U(:,k);
Z = Ured' * X;


%% Plot

axes('NextPlot','add','DataAspectRatio',[1 1 1],'XLim',[-5 5],'YLim',[0 eps],'Color','none');               %#   and don't use a background color
p1 = plot(Z,0,'r*','MarkerSize',10);
p2 = plot(0,0,'bo','MarkerSize',10);
xlabel({'z';'boundary: z = 0'})