function [Ktrain, Ktest, Ktest_self] = reference_kernel(Traindata, Testdata, basekernel, kernelparam, refoption, Negdata)
%Generalized Reference Kernel 
%
%Input:
%Traindata --> 'D x N' matrix of (positive) training vectors 
%Testdata --> 'D x Ntest' matrix of test vectors 
%basekernel --> function pointer to the basekernel function
%kernelparam --> hyperparameter for kernel (e.g., sigma for RBF)
%refoption --> 1-7, selected GRK approach
%Negdata --> 'D x Nneg' matrix of negative training vectors (only used for refoptions 5-7) 
%
%Output:
%Ktrain --> 'N x N' kernel matrix for training 
%Ktest --> 'N x Ntest' kernel matrix for testing
%Ktest_self --> '1 x Ntest' additional kernel vector for ksvdd (dd_tools) 

%Get the reference vectors correcponding to the selected GRK approach
M_ref = give_reference_vectors(Traindata, Negdata, refoption);

%Compute uncentered K_RR matrix and center it
K_RR_U = basekernel(M_ref, M_ref, kernelparam); 
N = size(K_RR_U, 1);
K_RR = (eye(N,N)-ones(N,N)/N) * K_RR_U * (eye(N,N)-ones(N,N)/N);

%Compute uncentered K_RX matrix and center it
[~, K_RX_U] = basekernel(M_ref, Traindata, kernelparam); 
NN = size(Traindata,2);
K_RX = (eye(N,N)-ones(N,N)/N) * (K_RX_U - (K_RR_U*ones(N,1)/N)*ones(1,NN));

%Compute uncentered K_RXtest matrix and center it
[~, K_RXtest_U] = basekernel(M_ref, Testdata, kernelparam); 
M = size(Testdata,2);  
K_RXtest = (eye(N,N)-ones(N,N)/N) * (K_RXtest_U - (K_RR_U*ones(N,1)/N)*ones(1,M));

%Compute the final Generalized Reference Kernels
[evcs,evls] = eig(K_RR);        evls = diag(evls);
[U, s] = sortEigVecs(evcs,evls);  S = diag(s);
LL = length(s);

Ktrain = K_RX' * U(:,1:LL) * S(1:LL,1:LL)^(-1) * U(:,1:LL)' * K_RX;
Ktest = K_RX' * U(:,1:LL) * S(1:LL,1:LL)^(-1) * U(:,1:LL)' * K_RXtest;
Phi_test = S(1:LL,1:LL)^(-0.5) * U(:,1:LL)'*K_RXtest;
Ktest_self = sum(Phi_test.^2,1); 
