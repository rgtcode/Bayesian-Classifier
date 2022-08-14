function [X Y]=white(x,y)
X=zeros(2,1);
Y=zeros(4);
a=zeros(4);
mn=zeros(2,1);
a=covmat(x,y);
[U V]=eig(a);
Aw=U*(V^(-0.5));
[mn1 mn2]=mean(x,y);
mn(1,1)=mn1;
mn(2,1)=mn2;
X=Aw'*mn;
Y=Aw'*a*Aw;
end