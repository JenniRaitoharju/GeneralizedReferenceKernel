function labels = GRK_oneclass(method, Traindata, Testdata, Negdata, basekernel, useNPT, useGRK, refoption, kernelparam, classifparam )
%Generalized Reference Kernel for One-Class Classification (SVDD/OCSVM)
%
%Input:
%method --> 'svdd'/'ocsvm'
%Traindata --> 'D x N' matrix of (positive) training vectors 
%Testdata --> 'D x Ntest' matrix of test vectors 
%Negdata --> 'D x Nneg' matrix of negative training vectors (only used for refoptions 5-7) 
%basekernel --> function pointer to the basekernel function
%useNPT --> boolean, NPT (true) or kernel (false) approach
%useGRK --> boolean, GRK (true) or original method (false)
%refoption --> 1-7, selected GRK approach
%kernelparam --> hyperparameter for kernel (e.g., sigma for RBF)
%classifparam --> hyperparamer for classificaion (C for SVDD/Nu for OCSVM)
%
%Output:
%labels --> 'Ntest x 1' vector of predicted labels for the test samples 

%Transform data using NPT or get the kernel matrix using the selected
%generalized reference kernel approach
if useNPT && ~useGRK %Original NPT
    [Traindata, Testdata] = NPT(Traindata, Testdata, basekernel, kernelparam);
elseif useNPT && useGRK %NPT with GRK
    [Traindata, Testdata] = reference_NPT(Traindata, Testdata, basekernel, kernelparam, refoption, Negdata);
elseif ~useNPT && ~useGRK %KOriginal kernel
    [Ktrain, Ktest, Ktest_self] = basekernel(Traindata, Testdata, kernelparam);
elseif ~useNPT && useGRK %Kernel with GRK
    [Ktrain, Ktest, Ktest_self] = reference_kernel(Traindata, Testdata, basekernel, kernelparam, refoption, Negdata);
end

N = size(Traindata,2);
Ntest = size(Testdata,2);

switch method
    case 'svdd'
        if useNPT %SVDD with NPT
            %train
            Model = svmtrain(ones(N,1), Traindata', ['-s 5 -t 0 -c ',num2str(classifparam)]);
            %test
            labels = svmpredict(ones(Ntest,1), Testdata', Model);
        else %SVDD with kernel
            %train
            Model = ksvdd(Ktrain, classifparam);
            %test
            w2 = ksvdd( [Ktest' Ktest_self'], Model, 'trained execution');
            tmpMat = +w2;  Dfrs = tmpMat(:,1)-tmpMat(:,2);  labels = sign(Dfrs);
        end                
    case 'ocsvm'
        if useNPT %OCSVM with NPT
            %train
            svm_model = svmtrain(ones(N,1), Traindata', sprintf('-t 0 -s 2 -n %f', classifparam));
            %test
            labels = svmpredict(ones(Ntest,1), Testdata', svm_model);
        else %OCSVM with kernel
            %Train
            Ktrain_svm =  [ (1:size(Ktrain,2))' , Ktrain' ];
            Ktest_svm =  [ (1:size(Ktest,2))' , Ktest' ];
            svm_model = svmtrain(ones(size(Ktrain,1),1), Ktrain_svm, sprintf('-t 4 -q -s 2 -n %f', classifparam));
            %Test
            labels = svmpredict(Testlabels, Ktest_svm, svm_model, '-q');
        end
end