function [Ktrain, Ktest, Ktest_self] = kernel_rbf(Traindata, Testdata, sigma)
%RBF kernel 
%
%Input:
%Traindata --> 'D x N' matrix of (positive) training vectors 
%Testdata --> 'D x Ntest' matrix of test vectors 
%sigma --> hyperparameter defining RBF kernel width
%
%Output:
%Ktrain --> 'N x N' kernel matrix for training 
%Ktest --> 'N x Ntest' kernel matrix for testing
%Ktest_self --> '1 x Ntest' additional kernel vector for ksvdd (dd_tools) 

N = size(Traindata,2);  Ntest = size(Testdata,2);

%Compute the distances 
Dtrain = ((sum(Traindata'.^2,2)*ones(1,N))+(sum(Traindata'.^2,2)*ones(1,N))'-(2*(Traindata'*Traindata)));
Dtest = ((sum(Traindata'.^2,2)*ones(1,Ntest))+(sum(Testdata'.^2,2)*ones(1,N))'-(2*(Traindata'*Testdata)));

%Compute kernel width using average distance of training samples
sigma2 = sigma * mean(mean(Dtrain));  gamma = 2.0 * sigma2;

%Compute kernel matrices 
Ktrain = exp(-Dtrain/(gamma));  
Ktrain = (Ktrain + Ktrain')/2;  
Ktest = exp(-Dtest/(gamma));
Ktest_self = ones(1, Ntest);
