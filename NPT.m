function [Phi_train,Phi_test] = NPT(Traindata, Testdata, basekernel, kernelparam)
%Non-linear Projection Trick
%
%Input:
%Traindata --> 'D x N' matrix of (positive) training vectors 
%Testdata --> 'D x Ntest' matrix of test vectors 
%
%Output:
%Phi_train --> 'M x N' Phi matrix, where M depends on Ktrain rank 
%Phi_test --> 'M x Ntest' Phi matrix, where M depends on Ktrain rank 

N = size(Traindata, 2); 
Ntest = size(Testdata, 2);

%Get uncentered matrices
[KtrainU, KtestU] = basekernel(Traindata, Testdata, kernelparam); 

%Center Ktrain
Ktrain = (eye(N,N)-ones(N,N)/N) * KtrainU * (eye(N,N)-ones(N,N)/N);

%Get Phi matrix using eigendecomposition
[evcs,evls] = eig(Ktrain);        evls = diag(evls);
[U, s] = sortEigVecs(evcs,evls);  S = diag(s);
Phi_train = S^(0.5) * U';

%Center Ktest and get Phi_test
Ktest = (eye(N,N)-ones(N,N)/N) * (KtestU - (KtrainU*ones(N,1)/N)*ones(1,Ntest));
Phi_test = S^(-0.5)*U'*Ktest; 
