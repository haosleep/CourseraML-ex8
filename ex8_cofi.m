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

%% =============== Part 1: Loading movie ratings dataset ================
%  You will start by loading the movie ratings dataset to understand the
%  structure of the data.
%  

% 這邊要練習的則是運用協同過濾算法執行的電影推薦系統
fprintf('Loading movie ratings dataset.\n\n');

%  Load data
% 先讀取所需的資料
% 內含Y,R兩個矩陣,格式都是1682x943
% Y表示1682部電影的943個用戶的評分,分數為1~5
% R則是區別用戶是否有對該電影評分,有評分是1沒評分是0
load ('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  From the matrix, we can compute statistics like average rating.
% 取得第一部電影的平均分數
fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', ...
        mean(Y(1, R(1, :))));

%  We can "visualize" the ratings matrix by plotting it with imagesc
% 將Y資料繪製出來
imagesc(Y);
ylabel('Movies');
xlabel('Users');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============ Part 2: Collaborative Filtering Cost Function ===========
%  You will now implement the cost function for collaborative filtering.
%  To help you debug your cost function, we have included set of weights
%  that we trained on that. Specifically, you should complete the code in 
%  cofiCostFunc.m to return J.

%  Load pre-trained weights (X, Theta, num_users, num_movies, num_features)
% 讀取另外需要的參數
% 格式1682x10 的X矩陣 表示1682部電影各自的特徵向量
% 格式943x10 的Theta矩陣 表示943個用戶各自的特徵向量
% 其他相關參數
% num_users 用戶數 = 943
% num_movies 電影數 = 1682
% num_features 特徵數 = 10
load ('ex8_movieParams.mat');

%  Reduce the data set size so that this runs faster
% 為了作業練習的執行速度可以加快,將相關的參數縮減
num_users = 4; num_movies = 5; num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

%  Evaluate cost function
% 在cofiCostFunc.m計算損失函數(part2作業)
% 作業設計是分階段性完成的
% 這邊先設最後一個參數(lambda)為0確認還沒加正則化部分時的損失函數結果是否正確
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
               num_features, 0);
           
fprintf(['Cost at loaded parameters: %f '...
         '\n(this value should be about 22.22)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============== Part 3: Collaborative Filtering Gradient ==============
%  Once your cost function matches up with ours, you should now implement 
%  the collaborative filtering gradient function. Specifically, you should 
%  complete the code in cofiCostFunc.m to return the grad argument.
%  
fprintf('\nChecking Gradients (without regularization) ... \n');

%  Check gradients by running checkNNGradients
% 利用隨機設定參數的方式檢查損失函數的程式是否正確
% checkCostFunction不傳其他參數時lambda會設定為0,可以先忽略掉正規化的部分
checkCostFunction;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ========= Part 4: Collaborative Filtering Cost Regularization ========
%  Now, you should implement regularization for the cost function for 
%  collaborative filtering. You can implement it by adding the cost of
%  regularization to the original cost computation.
%  

%  Evaluate cost function
% 接下來檢查加上正則化的損失函數(part4作業)
J = cofiCostFunc([X(:) ; Theta(:)], Y, R, num_users, num_movies, ...
               num_features, 1.5);
           
fprintf(['Cost at loaded parameters (lambda = 1.5): %f '...
         '\n(this value should be about 31.34)\n'], J);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ======= Part 5: Collaborative Filtering Gradient Regularization ======
%  Once your cost matches up with ours, you should proceed to implement 
%  regularization for the gradient. 
%

%  
fprintf('\nChecking Gradients (with regularization) ... \n');

%  Check gradients by running checkNNGradients
% 和part3相同進行檢查
checkCostFunction(1.5);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============== Part 6: Entering ratings for a new user ===============
%  Before we will train the collaborative filtering model, we will first
%  add ratings that correspond to a new user that we just observed. This
%  part of the code will also allow you to put in your own ratings for the
%  movies in our dataset!
%
% 讀取電影列表
movieList = loadMovieList();

%  Initialize my ratings
my_ratings = zeros(1682, 1);

% Check the file movie_idx.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
% 對各電影給值評分
my_ratings(1) = 4;

% Or suppose did not enjoy Silence of the Lambs (1991), you can set
my_ratings(98) = 2;

% We have selected a few movies we liked / did not like and the ratings we
% gave are as follows:
my_ratings(7) = 3;
my_ratings(12)= 5;
my_ratings(54) = 4;
my_ratings(64)= 5;
my_ratings(66)= 3;
my_ratings(69) = 5;
my_ratings(183) = 4;
my_ratings(226) = 5;
my_ratings(355)= 5;

% 把有評分的分數和對應的電影名print出來
fprintf('\n\nNew user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ================== Part 7: Learning Movie Ratings ====================
%  Now, you will train the collaborative filtering model on a movie rating 
%  dataset of 1682 movies and 943 users
%

fprintf('\nTraining collaborative filtering...\n');

%  Load data
% 再讀取一次已有的資料庫Y,R
load('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  Add our own ratings to the data matrix
% 將part6自己評分的部分也加進Y,R矩陣了(增加一個用戶的資料)
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
% 對Y矩陣進行標準化(R是用來辨別有沒有評分)
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
% 設定相關參數
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
% 先隨機給定初始的X,Theta
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
% 設定機器學習fmincg需要的相關參數(梯度設置,迭代次數)
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
% 進行機器學習取得適合的theta
lambda = 10;
theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
% 將theta拆回成對應電影的特徵向量和對應用戶的特徵向量
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

fprintf('Recommender system learning completed.\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================== Part 8: Recommendation for you ====================
%  After training the model, you can now make recommendations by computing
%  the predictions matrix.
%

% 取得預測結果
p = X * Theta';
% 因為part7時是將part6的評分加在最左行
% 這邊把屬於自己評分的部分再取回來
my_predictions = p(:,1) + Ymean;

% 讀取電影列表
movieList = loadMovieList();

% 將結果進行排序(descend表由大到小的降冪排列)
% r表示值,ix表示位置
[r, ix] = sort(my_predictions, 'descend');
% 將前10部推薦電影和推薦分數列出
fprintf('\nTop recommendations for you:\n');
for i=1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), ...
            movieList{j});
end

% 這邊則列出part6時所輸入的原本評分資訊
fprintf('\n\nOriginal ratings provided:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end
