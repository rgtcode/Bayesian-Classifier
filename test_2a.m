clc;
clear all;
close all;

plot_flag = 1;

dataDir = 'data/LinearlySeperable/12';
% dataDir = 'data/RealData/12';

train_data_file = strcat(dataDir,'/trian.txt');
dev_data_file   = strcat(dataDir,'/dev.txt');

fid = fopen(train_data_file,'r');
tt = textscan(fid,'%f, %f, %d');
fclose(fid);

fid = fopen(dev_data_file,'r');
dd = textscan(fid,'%f, %f, %d');
fclose(fid);

nSamples = length(tt{1}); % Total number of training samples
[~,ddim] = size(tt); % Data Dimensions
ddim = ddim - 1; % Remove the labels from data
nClasses = 3; % Number of Classes

% Setup Data and Labels
tdata = zeros(length(tt{1}), ddim);
ddata = zeros(length(dd{1}), ddim);
for i = 1:ddim
    tdata(:,i) = tt{i}(:);
    ddata(:,i) = dd{i}(:);
end
tlabels = [tt{1,3}(:)];
dlabels = [dd{1,3}(:)];


% Case 1: Bayes with Covariance same for all classes
% Without Loss of Generality, we make the following assumptions
% Assumption 1: The class probabilities P_wi's are equal for all classes
% Assumption 2: We choose the covaiance of class 1 for the remaining classes
% Assumption 3: The data distribution for each class follows a normal dist

% Compute P_w_i
for i = 1:nClasses
    P_w(i) = 1/nClasses;
end


% Compute parameters of class conditional pdf p_x_given_w_i for each class
% separately
for i = 1:nClasses
    [m(:,i),C(:,:,i)] = GetMeanCov(tdata, tlabels, i);
end

% Compute the Class Conditional PDFs for each class
minsdVal = sqrt(abs(min(C(:))));
maxsdVal = sqrt(abs(max(C(:))));
minMeanVal = min(m(:));
maxMeanVal = max(m(:));

%


x1Range = minMeanVal - 8*minsdVal:0.50:maxMeanVal + 8*maxsdVal;
for i = 1:nClasses
    x1_ind = 1;
    for x1 = x1Range
        x2_ind = 1;
        for x2 = x1Range
            p_x_given_w_i(x1_ind,x2_ind,i) = GetNDGaussian([x1;x2], m(:,i), C(:,:,1)); % Using Same Cov matrix for all classes
            x2_ind = x2_ind + 1;
        end
        x1_ind = x1_ind + 1;
    end
end

% Plot 1: The Gaussian PDFs for each class
if (plot_flag)
    figure; surf(x1Range, x1Range, p_x_given_w_i(:,:,1));
    hold on;surf(x1Range, x1Range, p_x_given_w_i(:,:,2));
    hold on;surf(x1Range, x1Range, p_x_given_w_i(:,:,3));
end

% Compute Decision Boundary
x1_ind = 1;
for x1 = x1Range
    x2_ind = 1;
    for x2 = x1Range
        [val,ind] = max(p_x_given_w_i(x1_ind, x2_ind, :));
        db(x1_ind, x2_ind) = ind^4;
        x2_ind = x2_ind + 1;
    end
    x1_ind = x1_ind + 1;
end

% Plot 2: Plot the decision boundary
if (plot_flag)
    hold on;imagesc(x1Range, x1Range, db);
end

% Plot 2b: Superimpose training data
if (plot_flag)
    hold on;scatter(tdata(:,2), tdata(:,1),'w');
end

% % Plot 2c: Superimpose dev data
% if (plot_flag)
%     hold on;scatter(ddata(:,2), ddata(:,1),'w');
% end

% Compute Classification Accuracy on Development Data
% Neglecting the denoninator as it is not needed for computing max in case
% of Bayes classification decision
p_w_i_given_x = double(zeros(nClasses,1));
for j=1:length(ddata)
    for i = 1:nClasses
        p_x_given_w_i_hat = GetNDGaussian(ddata(j,:)', m(:,i), C(:,:,1));
        p_w_i_given_x(i) = p_x_given_w_i_hat * P_w(i);
    end
    [val,ind] = max(p_w_i_given_x);
    dlabels_hat(j,1) = ind;
end

n_misclassified = sum(abs(int32(dlabels_hat) - dlabels));
accuracy = (length(ddata) - n_misclassified)*100/length(ddata);

% Plot 3: Constant Density Curve and Eigen Vectors for Training Data
% figure;scatter(tdata(:,2), tdata(:,1),'b');    
figure; contour(x1Range, x1Range, p_x_given_w_i(:,:,1));
hold on;contour(x1Range, x1Range, p_x_given_w_i(:,:,2));
hold on;contour(x1Range, x1Range, p_x_given_w_i(:,:,3));

for i = 1:nClasses
    cInd = find(tlabels==i);
    A_mat = tdata(cInd,:);
    [V, D] = eig(A_mat'*A_mat);
    v1 = V(:,1);
    v2 = V(:,2);
    d = sqrt(diag(D));

    hold on;
    quiver(m(2,i),m(1,i),v1(1),v2(1),d(1)/15,'k','LineWidth',3,'Color','r');
    quiver(m(2,i),m(1,i),v1(2),v2(2),d(2)/15,'k','LineWidth',3,'Color','r');
end
colorbar;
