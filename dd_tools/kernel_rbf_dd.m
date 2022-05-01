function [Ktrain, Ktest, A] = kernel_rbf_dd(M_train, M_test, normalize, M_ref, gamma)
%Name changed to avoid using this version

N = size(M_train,2);  NN = size(M_test,2);
if nargin < 3,  normalize = 0;  end

if nargin < 4

    if normalize==1  % center  and normalize data
        
        mean_train = mean(M_train,2);
        M_train = M_train - mean_train*ones(1,size(M_train,2));
        M_test = M_test - mean_train*ones(1,size(M_test,2));
        norms_train = diag(M_train' * M_train) .^ (0.5);
        M_train = M_train ./ (ones(size(M_train,1),1)*norms_train');
        norms_test = diag(M_test' * M_test) .^ (0.5);
        M_test = M_test ./ (ones(size(M_test,1),1)*norms_test');
    end

    Dtrain = ((sum(M_train'.^2,2)*ones(1,N))+(sum(M_train'.^2,2)*ones(1,N))'-(2*(M_train'*M_train)));
    Dtest = ((sum(M_train'.^2,2)*ones(1,NN))+(sum(M_test'.^2,2)*ones(1,N))'-(2*(M_train'*M_test)));
    A = sqrt(mean(mean(Dtrain)));
else
    
    if normalize==1  % center  and normalize data
        
        mean_train = mean(M_train,2);
        M_train = M_train - mean_train*ones(1,size(M_train,2));
        M_test = M_test - mean_train*ones(1,size(M_test,2));
        norms_train = diag(M_train' * M_train) .^ (0.5);
        M_train = M_train ./ (ones(size(M_train,1),1)*norms_train');
        norms_test = diag(M_test' * M_test) .^ (0.5);
        M_test = M_test ./ (ones(size(M_test,1),1)*norms_test');
        
        mean_ref = mean(M_ref,2);
        M_ref = M_ref - mean_ref*ones(1,size(M_ref,2));
        norms_ref = diag(M_ref' * M_ref) .^ (0.5);
        M_ref = M_ref ./ (ones(size(M_ref,1),1)*norms_ref');
    end
    
    MM = size(M_ref,2);
    Dtrain = ((sum(M_ref'.^2,2)*ones(1,N))+(sum(M_train'.^2,2)*ones(1,MM))'-(2*(M_ref'*M_train)));
    Dtest = ((sum(M_ref'.^2,2)*ones(1,NN))+(sum(M_test'.^2,2)*ones(1,MM))'-(2*(M_ref'*M_test)));
    A = sqrt(mean(mean(Dtrain)));
end

if gamma==0
    Ktrain = exp(-Dtrain/(A*A));          Ktest = exp(-Dtest/(A*A));
else
    Ktrain = exp(-Dtrain/(gamma*gamma));  Ktest = exp(-Dtest/(gamma*gamma));
end
