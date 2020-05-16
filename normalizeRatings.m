function [Ynorm, Ymean] = normalizeRatings(Y, R)
%NORMALIZERATINGS Preprocess data by subtracting mean rating for every 
%movie (every row)
%   [Ynorm, Ymean] = NORMALIZERATINGS(Y, R) normalized Y so that each movie
%   has a rating of 0 on average, and returns the mean rating in Ymean.
%

[m, n] = size(Y);
Ymean = zeros(m, 1);
Ynorm = zeros(size(Y));
for i = 1:m
    % 先用R區分出有效(有評分的)資料
    idx = find(R(i, :) == 1);
    % 計算平均值
    Ymean(i) = mean(Y(i, idx));
    % 進行標準化
    Ynorm(i, idx) = Y(i, idx) - Ymean(i);
end

end
