%% Machine Learning Online Class
%  Exercise 8 | Anomaly Detection and Collaborative Filtering
%
%  Instructions
%  ------------
%
%  This file contains code that helps you get started on the
%  exercise. You will need to complete the following functions:
%
%     estimateGaussian.m
%     selectThreshold.m
%     cofiCostFunc.m
%
%  For this exercise, you will not need to change any code in this file,
%  or any other files other than those mentioned above.
%

%% Initialization
clear ; close all; clc

%% ================== Part 1: Load Example Dataset  ===================
%  We start this exercise by using a small dataset that is easy to
%  visualize.
%
%  Our example case consists of 2 network server statistics across
%  several machines: the latency and throughput of each machine.
%  This exercise will help us find possibly faulty (or very fast) machines.
%

fprintf('Visualizing example dataset for outlier detection.\n\n');

%  The following command loads the dataset. You should now have the
%  variables X, Xval, yval in your environment

% 本次作業要練習的是異常檢測
% 先讀取用以測試的資料
% 格式 307x2 的X和Xval表示 伺服器的吞吐量和延遲時間的 兩個特徵的307筆資料
% 格式 307x1 的yval則是對應Xval是否是異常的伺服器
load('ex8data1.mat');

%  Visualize the example dataset
% 先要用到的是X,將資料結果繪製成二維圖
plot(X(:, 1), X(:, 2), 'bx');
axis([0 30 0 30]);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Program paused. Press enter to continue.\n');
pause


%% ================== Part 2: Estimate the dataset statistics ===================
%  For this exercise, we assume a Gaussian distribution for the dataset.
%
%  We first estimate the parameters of our assumed Gaussian distribution, 
%  then compute the probabilities for each of the points and then visualize 
%  both the overall distribution and where each of the points falls in 
%  terms of that distribution.
%
fprintf('Visualizing Gaussian fit.\n\n');

%  Estimate my and sigma2
% 接著要計算高斯分布所對應的參數
% 運用estimateGaussian.m求得均值μ和方差σ^2 (part2作業)
[mu sigma2] = estimateGaussian(X);

%  Returns the density of the multivariate normal at each data point (row) 
%  of X
% 取得高斯分布的結果
p = multivariateGaussian(X, mu, sigma2);

%  Visualize the fit
% 將高斯分布的輪廓繪製成二維圖
visualizeFit(X,  mu, sigma2);
xlabel('Latency (ms)');
ylabel('Throughput (mb/s)');

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================== Part 3: Find Outliers ===================
%  Now you will find a good epsilon threshold using a cross-validation set
%  probabilities given the estimated Gaussian distribution
% 

% 這次則是使用驗證集Xval的資料來取得高斯分布結果
pval = multivariateGaussian(Xval, mu, sigma2);

% 將前面求得的pval和已準備好的實際結果yval傳入selectThreshold.m
% 選擇出最好的判斷門檻值和其F1 score (part3作業)
[epsilon F1] = selectThreshold(yval, pval);
fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 8.99e-05)\n');
fprintf('   (you should see a Best F1 value of  0.875000)\n\n');

%  Find the outliers in the training set and plot the
% 前面使用Xval和yval取得了適合的門檻值後
% 用這門檻值和part2時的p比較來判斷X的資料中哪些是異常的
outliers = find(p < epsilon);

%  Draw a red circle around those outliers
% 將判斷為異常的資料畫上稍大的紅圈(ro)當標註
hold on
plot(X(outliers, 1), X(outliers, 2), 'ro', 'LineWidth', 2, 'MarkerSize', 10);
hold off

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================== Part 4: Multidimensional Outliers ===================
%  We will now use the code from the previous part and apply it to a 
%  harder problem in which more features describe each datapoint and only 
%  some features indicate whether a point is an outlier.
%

%  Loads the second dataset. You should now have the
%  variables X, Xval, yval in your environment

% 接著測試高維的資料(比較貼近現實的狀況)
% 格式 1000x11 的X矩陣 表示11個特徵值的1000筆資料
% 格式 100x11 的Xval矩陣則是同樣11個特徵值,用於決定門檻值的100筆驗證集資料
% 格式 100x1 的yval矩陣是對應Xval,是否異常的結果
load('ex8data2.mat');

% 下面的處理步驟其實就跟part2,part3一樣

%  Apply the same steps to the larger dataset
[mu sigma2] = estimateGaussian(X);

%  Training set 
p = multivariateGaussian(X, mu, sigma2);

%  Cross-validation set
pval = multivariateGaussian(Xval, mu, sigma2);

%  Find the best threshold
[epsilon F1] = selectThreshold(yval, pval);

fprintf('Best epsilon found using cross-validation: %e\n', epsilon);
fprintf('Best F1 on Cross Validation Set:  %f\n', F1);
fprintf('   (you should see a value epsilon of about 1.38e-18)\n');
fprintf('   (you should see a Best F1 value of 0.615385)\n');
fprintf('# Outliers found: %d\n\n', sum(p < epsilon));
