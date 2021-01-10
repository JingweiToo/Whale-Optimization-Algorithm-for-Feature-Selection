%---------------------------------------------------------------------%
%  Whale Optimization Algorithm (WOA) source codes demo version       %
%---------------------------------------------------------------------%


%---Inputs-----------------------------------------------------------
% feat     : feature vector ( Instances x Features )
% label    : label vector ( Instances x 1 )
% N        : Number of whales
% max_Iter : Maximum number of iterations
% b        : Constant 

%---Output-----------------------------------------------------------
% sFeat    : Selected features (instances x features)
% Sf       : Selected feature index
% Nf       : Number of selected features
% curve    : Convergence curve
%--------------------------------------------------------------------


%% Whale Optimization Algorithm
clc, clear, close; 
% Benchmark data set 
load ionosphere.mat;

% Set 20% data as validation set
ho = 0.2; 
% Hold-out method
HO = cvpartition(label,'HoldOut',ho);

% Parameter setting
N        = 10; 
max_Iter = 100; 

% Whale Optimization Algorithm
[sFeat,Sf,Nf,curve] = jWOA(feat,label,N,max_Iter,HO);

% Accuracy
Acc = jKNN(sFeat,label,HO); 
fprintf('\n Accuracy: %g %%',Acc);

% Plot convergence curve
plot(1:max_Iter,curve); 
xlabel('Number of Iterations');
ylabel('Fitness Value');
title('WOA'); grid on;




