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

% �o��n�m�ߪ��h�O�B�Ψ�P�L�o��k���檺�q�v���˨t��
fprintf('Loading movie ratings dataset.\n\n');

%  Load data
% ��Ū���һݪ����
% ���tY,R��ӯx�},�榡���O1682x943
% Y���1682���q�v��943�ӥΤ᪺����,���Ƭ�1~5
% R�h�O�ϧO�Τ�O�_����ӹq�v����,�������O1�S�����O0
load ('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies on 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  From the matrix, we can compute statistics like average rating.
% ���o�Ĥ@���q�v����������
fprintf('Average rating for movie 1 (Toy Story): %f / 5\n\n', ...
        mean(Y(1, R(1, :))));

%  We can "visualize" the ratings matrix by plotting it with imagesc
% �NY���ø�s�X��
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
% Ū���t�~�ݭn���Ѽ�
% �榡1682x10 ��X�x�} ���1682���q�v�U�۪��S�x�V�q
% �榡943x10 ��Theta�x�} ���943�ӥΤ�U�۪��S�x�V�q
% ��L�����Ѽ�
% num_users �Τ�� = 943
% num_movies �q�v�� = 1682
% num_features �S�x�� = 10
load ('ex8_movieParams.mat');

%  Reduce the data set size so that this runs faster
% ���F�@�~�m�ߪ�����t�ץi�H�[��,�N�������Ѽ��Y��
num_users = 4; num_movies = 5; num_features = 3;
X = X(1:num_movies, 1:num_features);
Theta = Theta(1:num_users, 1:num_features);
Y = Y(1:num_movies, 1:num_users);
R = R(1:num_movies, 1:num_users);

%  Evaluate cost function
% �bcofiCostFunc.m�p��l�����(part2�@�~)
% �@�~�]�p�O�����q�ʧ�����
% �o����]�̫�@�ӰѼ�(lambda)��0�T�{�٨S�[���h�Ƴ����ɪ��l����Ƶ��G�O�_���T
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
% �Q���H���]�w�Ѽƪ��覡�ˬd�l����ƪ��{���O�_���T
% checkCostFunction���Ǩ�L�ѼƮ�lambda�|�]�w��0,�i�H�����������W�ƪ�����
checkCostFunction;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ========= Part 4: Collaborative Filtering Cost Regularization ========
%  Now, you should implement regularization for the cost function for 
%  collaborative filtering. You can implement it by adding the cost of
%  regularization to the original cost computation.
%  

%  Evaluate cost function
% ���U���ˬd�[�W���h�ƪ��l�����(part4�@�~)
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
% �Mpart3�ۦP�i���ˬd
checkCostFunction(1.5);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ============== Part 6: Entering ratings for a new user ===============
%  Before we will train the collaborative filtering model, we will first
%  add ratings that correspond to a new user that we just observed. This
%  part of the code will also allow you to put in your own ratings for the
%  movies in our dataset!
%
% Ū���q�v�C��
movieList = loadMovieList();

%  Initialize my ratings
my_ratings = zeros(1682, 1);

% Check the file movie_idx.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
% ��U�q�v���ȵ���
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

% �⦳���������ƩM�������q�v�Wprint�X��
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
% �AŪ���@���w������ƮwY,R
load('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  Add our own ratings to the data matrix
% �Npart6�ۤv�����������]�[�iY,R�x�}�F(�W�[�@�ӥΤ᪺���)
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
% ��Y�x�}�i��зǤ�(R�O�Ψӿ�O���S������)
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
% �]�w�����Ѽ�
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
% ���H�����w��l��X,Theta
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
% �]�w�����ǲ�fmincg�ݭn�������Ѽ�(��׳]�m,���N����)
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
% �i������ǲߨ��o�A�X��theta
lambda = 10;
theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
% �Ntheta��^�������q�v���S�x�V�q�M�����Τ᪺�S�x�V�q
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

% ���o�w�����G
p = X * Theta';
% �]��part7�ɬO�Npart6�������[�b�̥���
% �o����ݩ�ۤv�����������A���^��
my_predictions = p(:,1) + Ymean;

% Ū���q�v�C��
movieList = loadMovieList();

% �N���G�i��Ƨ�(descend��Ѥj��p�������ƦC)
% r��ܭ�,ix��ܦ�m
[r, ix] = sort(my_predictions, 'descend');
% �N�e10�����˹q�v�M���ˤ��ƦC�X
fprintf('\nTop recommendations for you:\n');
for i=1:10
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), ...
            movieList{j});
end

% �o��h�C�Xpart6�ɩҿ�J���쥻������T
fprintf('\n\nOriginal ratings provided:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end
