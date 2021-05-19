function [Pn,Tn]=devide_blocks(P,T,mini_batch)
% P: intputs
% T: targets
% mini_batch :mini batche size

% number of trainig data
nTrainingData=size(P,1);
% dividion process 
c=0;
for n = 1 : mini_batch : nTrainingData
    if (n+mini_batch-1) < nTrainingData
    c=c+1;
    Pn{c} = P(n:(n+mini_batch-1),:);      
    Tn{c} = T(n:(n+mini_batch-1),:);
    end
end