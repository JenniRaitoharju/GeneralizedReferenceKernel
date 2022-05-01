function [sevcs, sevls] = sortEigVecs(evcs, evls, type)
%Sort eigenvalues and vectors
%Output:
%sevls --> row vector of sorted eigenvalues
%sevcs --> matrix of sorted eigenvectors as columns
%Input:
%evls --> row vector of eigenvalues
%evcs --> matrix of corresponding eigenvectors as columns
%type --> 'descend' (default)/'ascend'

if nargin < 3,  type = 'descend';  end

%Clean
evls(imag(evls)~=0) = 0; 
evcs(imag(evcs)~=0) = real(evcs(imag(evcs)~=0));    
evls(isinf(evls))=0.0; 
evls(isnan(evls))=0.0; 
evcs(isinf(evcs))=0.0; 
evcs(isnan(evcs))=0.0;

%Round near-zero eigenvalues to zero
evls(evls<10^-6) = 0.0;

%Sort eigenvalues and remove zeros
[auxv, auxi] = sort(evls, type);
sevls = auxv;
sevcs = evcs(:, auxi);
sevcs = sevcs(:, sevls > 0);
sevls = sevls(sevls > 0);


