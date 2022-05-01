function M_ref = give_reference_vectors(Traindata, Negdata, refoption)
%Give reference vectors according to the selected reference option
%
%Input:
%Traindata --> 'D x N' matrix of (positive) training vectors
%Negdata --> 'D x Nneg' matrix of negative training vectors (only used for refoptions 5-7) 
%refoption --> 1-7, selected GRK approach
%
%Output:
%M_ref --> 'D x M' matrix of reference vectors, where M depends on the selected reference option

[D,N] = size(Traindata); 
switch refoption
    case 1 %Give all N original training vectors
        M_ref = Traindata;  
    case 2 %Give N random vectors 
        M_ref = randn(D, N);
    case 3 %Randomly select N/2 training vectors
        n = floor(N/2);
        randIdcs = randperm(N, n);
        M_ref = Traindata(:,randIdcs);
    case 4 %Give N/2 random vectors
        n = floor(N/2);
        M_ref = randn(D, n);
    case 5 %Give N training vectors and min(N,Nneg) negative samples
        Nneg = size(Negdata,2);
        M = min(N, Nneg);
        if M < Nneg
            randIdcs = randperm(Nneg, M);
            Negdata = Negdata(:,randIdcs);
        end
        M_ref = [Traindata, Negdata];
    case 6 %Give N training vectors and min(N,Nneg) random vectors
        Nneg = size(Negdata,2);
        M = min(N, Nneg);
        Randdata = randn(D, M);
        M_ref = [Traindata, Randdata];
    case 7 %Give N + min(N,Nneg) random vectors
        Nneg = size(Negdata,2);
        M = min(N, Nneg);
        M_ref = randn(D, N+M);
end
 