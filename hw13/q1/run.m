clear all
close all
clc

x1 = [-1 0] ;
x2 = [1];

axes('NextPlot','add','DataAspectRatio',[1 1 1],'XLim',[-2 2],'YLim',[0 eps],'Color','none');
p1 = plot(x1,0,'r*','MarkerSize',10);
p2 = plot(x2,0,'b*','MarkerSize',10);
p3 = plot(0.5,0,'go','MarkerSize',10);
xlabel({'x';'boundary: x = 0.5'})