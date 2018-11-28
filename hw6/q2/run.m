clear all;
close all;
clc;

x = -10:0.1:10;
y1 = normpdf(x,1,1.414);
y2 = normpdf(x,3,1);

figure
title('Conditional densities')
hold on
plot(x,y1)
plot(x,y2)
xlabel('x')
ylabel('p(x/w)')
legend('p(x/w1)','p(x/w2)')

evidence = y1 * 0.6 + y2 * 0.4;
posterior1 = y1 * 0.6 ./ evidence;
posterior2 = y2 * 0.4 ./ evidence;

figure
title('Posterior densities')
hold on
plot(x,posterior1)
plot(x,posterior2)
xlabel('x')
ylabel('p(w/x)')
legend('p(w1/x)','p(w2/x)')