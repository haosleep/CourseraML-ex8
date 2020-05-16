function [bestEpsilon bestF1] = selectThreshold(yval, pval)
%SELECTTHRESHOLD Find the best threshold (epsilon) to use for selecting
%outliers
%   [bestEpsilon bestF1] = SELECTTHRESHOLD(yval, pval) finds the best
%   threshold to use for selecting outliers based on the results from a
%   validation set (pval) and the ground truth (yval).
%

bestEpsilon = 0;
bestF1 = 0;
F1 = 0;

% 決定用以判斷的門檻值(從最小值~最大值共1001組)
stepsize = (max(pval) - min(pval)) / 1000;
for epsilon = min(pval):stepsize:max(pval)
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Compute the F1 score of choosing epsilon as the
    %               threshold and place the value in F1. The code at the
    %               end of the loop will compare the F1 score for this
    %               choice of epsilon and set it to be the best epsilon if
    %               it is better than the current choice of epsilon.
    %               
    % Note: You can use predictions = (pval < epsilon) to get a binary vector
    %       of 0's and 1's of the outlier predictions

    % 若高斯分布對應值小於門檻值則判斷為異常
    predictions = (pval < epsilon);
    % 計算F1 score
    % tp:判斷異常,實際異常
    % fp:判斷異常,實際正常
    % fn:判斷正常,實際異常
    tp = sum((predictions == 1) & (yval == 1));
    fp = sum((predictions == 1) & (yval == 0));
    fn = sum((predictions == 0) & (yval == 1));
    
    % 因為判斷的門檻值會從pval最小值開始
    % 當門檻是pval最小值時,自然不會有任何資料被判斷異常=> tp和fp都會是0
    % prec會發生division by zero的問題
    % 不過基本能無視
    % (所有資料都判斷正常的門檻值,F1 score會是0,不會是適合的門檻值)
    % 況且實際運用時把for迴圈規則改一下就能避開問題了
    prec = tp / (tp + fp);
    rec = tp / (tp + fn);

    F1 = (2 * prec * rec) / (prec + rec);


    % =============================================================
    
    % 當F1 score比當前紀錄的F1 score還高
    % 更新最高紀錄且紀錄此時的門檻值
    if F1 > bestF1
       bestF1 = F1;
       bestEpsilon = epsilon;
    end
end

end
