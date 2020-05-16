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
    % ����R�Ϥ��X����(��������)���
    idx = find(R(i, :) == 1);
    % �p�⥭����
    Ymean(i) = mean(Y(i, idx));
    % �i��зǤ�
    Ynorm(i, idx) = Y(i, idx) - Ymean(i);
end

end
