%This files contains a simple demo for applying the Generalized Reference
%Kernel with SVDD/OCSVM on already preprocessed datasets

%Add libraries
addpath('prtools');
addpath('dd_tools');
addpath('libSVMmex');

%Add datasets
addpath('Datasets');

%Load a preprocessed dataset and select data split
%Note that training data includes also negative samples
dataset = 'Datasets/iris_targetclass_1'; % Preprocessed dataset (See 'Datasets/AboutData.txt')
datasplit = 1; %Each dataset has 5 different splits into train/test, select 1-5
load (dataset); 
Traindata=traindata5sets{1, datasplit};
Trainlabels=trainlabels5sets{1, datasplit};
Testdata=testdata5sets{1, datasplit};
Testlabels=testlabels5sets{1, datasplit};

%Select only positive data for training
Negdata=Traindata(:, Trainlabels==-1); %Negative training data is needed for reference kernel options 5-7
Traindata=Traindata(:, Trainlabels==1);

%Define experimental setup and set hyperparameter values
%See 'GRK_oneclass.m' for parameter definitions and
%'give_reference_vectors.m' for GRK variant definitions
method = 'svdd';
useNPT = true; 
useGRK = true; 
refoption = 5; 
basekernel = @kernel_rbf;
sigma = 1.0;
c = 0.5; 

%Run experiment
labels = GRK_oneclass( method, Traindata, Testdata, Negdata, basekernel, useNPT, useGRK, refoption, sigma, c );

%Evaluate performance metrics
results = evaluate(Testlabels,labels)


