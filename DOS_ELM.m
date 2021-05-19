function [net]=DOS_ELM(Trinputs,Tsinputs,Troutputs,Tsoutputs,Opts)

% Adaptive Dnoising Online sequential ELM 

% Inputs    :

% Trinputs  : training inputs (instances,features)
% Tsinputs  : testing inputs  (instances,features)
% Trtargets : training inputs (instances,features)
% Tstargets : testing inputs  (instances,features)
% Opts      : training Options

% Outputs   :  

% net:results of training
%
% Please cite this work as:
% ~~~~~~~~~~~~~~~~
%           T.Berghout,  L.H.Mouss,  O.Kadri,  L.Saïdi,  M.Benbouzid,(2020), 
%           " Aircraft engines Remaining  Useful Life  prediction  with  an 
%           adaptive denoising online sequential Extreme Learning Machine",
%           Engineering Applications of Artificial Intelligence,
%           Vol: 96, Issue:103936, 10.1016/j.engappai.2020.103936.
% ~~~~~~~~~~~~~~~~~
%% example how to use
% copy this into anothor file and run it (same directory)
% % Training Options 
% Options.mini_batch=10; % minibatch size
% Options.activF='relu'; % activation function ('relu','sin','tribas','hardlim','radbas')
% Options.Neurons=[100]; % number of neurons
% Options.lamdaMin=0.98; % forgetting parameter
% Options.mu=0.001;      % velosity parameter
% Options.C=100;         % regularization parameter
% Options.N_ratio=0.001; % Noise ratio
% Options.dB=0.09;       % Noise magnitude
% 
% [net]=DOS_ELM(Trinputs,Tsinputs,Troutputs,Tsoutputs,Opts)

%% Load Options
mini_batch=Opts.mini_batch; % mini_batch size
activF=Opts.activF;         % Activation function
Neurons=Opts.Neurons;       % Number of neurons
mu=Opts.mu;                 % sensitivity factor
lamdaMin=Opts.lamdaMin;     % initial forgetting factor
N_ratio=Opts.N_ratio;       % noise ration
dB=Opts.dB;                 % noise power

%% divide data
[Trinputs,Troutputs]=devide_blocks(Trinputs,Troutputs,mini_batch);
[Tsinputs,Tsoutputs]=devide_blocks(Tsinputs,Tsoutputs,mini_batch);
%% Initial phase
start_time_train=cputime;
P = Trinputs{1};% initial inputs
T = Troutputs{1};% initial targets
%% corrupt the input
P=corrupt(P,N_ratio,dB);
Cr{1}=P;
%% end corruptio
% initial mini-batch
% generate a random input weights
IW=rand(Neurons(1),size(P,2)); 
% calculating the temporal hidden layer  
tempH=IW*P';                   
% Activation function
 switch lower(activF)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H = 1 ./ (1 + exp(-tempH));
    case {'sin','sine'}
        %%%%%%%% Sine
        H = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H = double(hardlim(tempH));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H = tribas(tempH);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H = radbas(tempH);
    case {'relu'}
        %%%%%%%% linear function
        H = max(tempH,0);
% More activation functions can be added here                
 end        
clear IW 
B_AE=pinv(H') * P ;  % output weights of AE
B_OS=pinv(H') * T;   % output weights of OS_ELM
% initial variace matrix
M_AE = pinv(H * H');    % calculate the variance Matrix for the hidden layer
M_OS = pinv(H * H');    % calculate the variance Matrix for the hidden layer
% initial variance matrix
E{1}=P - scaledata((H' * B_AE),min(P(:)),max(P(:)));% initial error for the AE
En{1}=T - scaledata((H' * B_OS),min(T),max(T));% initial error of OS_ELM
% RMSE of prediction
e_AE(1,1)=sqrt(mse(E{1}));% RMSE
e_OS(1,1)=sqrt(mse(En{1}));% RMSE
lamdas_AE(1,1)=lamdaMin;
lamdas_OS(1,1)=lamdaMin;
%% Updating phase
c=0;% initialize a counter 
if numel(Trinputs)>1%                   
for t=2:numel(Trinputs)

Pnew = Trinputs{t};
Pnew=corrupt(Pnew,N_ratio,dB);
Cr{t}=Pnew;
Tnew = Troutputs{t};
Hnew_temp=(pinv(B_AE'))*Pnew'; % temporal hidden layer
Hnew=Hnew_temp;
%%%%%Activation function%%%
 switch lower(activF)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        Hnew = 1 ./ (1 + exp(-Hnew_temp));
    case {'sin','sine'}
        %%%%%%%% Sine
        Hnew = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        Hnew = double(hardlim(Hnew_temp));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        Hnew = tribas(Hnew_temp);
    case {'radbas'}
        %%%%%%%% Radial basis function
        Hnew = radbas(Hnew_temp);
    case {'relu'}
        %%%%%%%% linear function
        Hnew = max(Hnew_temp,0);
        %%%%%%%% More activation functions can be added here                
end
% error
E{t}= Pnew - scaledata((Hnew' * B_AE),min(Pnew(:)),max(Pnew(:)));   % error of AE
En{t}=Tnew - scaledata((Hnew' * B_OS),min(Tnew(:)),max(Tnew(:)));   %  error of OS_ELM
% RMSE
e_AE(t,1)=sqrt(mse(E{t}));% RMSE
e_OS(t,1)=sqrt(mse(En{t}));% RMSE


if e_OS(t,1)>e_OS(t-1,1)% if USS
c=c+1;
Index(c,1)=t;
% forgetting factor
% update lamda (dynamicly  forget old data)
lamda_AE=lamdaMin+(1-lamdaMin)*exp(-mu*sqrt(mse(E{t})));
lamda_OS=lamdaMin+(1-lamdaMin)*exp(-mu*sqrt(mse(En{t})));
% boundary constraints Adjustement for lamda_AE (forgeting factor)
if lamda_AE<=lamdaMin
lamda_AE=lamdaMin;
elseif lamda_AE>=1
lamda_AE=1;    
end
if lamda_OS<=lamdaMin
lamda_OS=lamdaMin;
elseif lamda_OS>=1
lamda_OS=1;    
end

% gain matrix
K_AE =(M_AE * Hnew) *((lamda_AE+eye(size(Pnew,1))+Hnew' * M_AE * Hnew)^(-1));% update gain matrix AE
K_OS =(M_OS * Hnew) *((lamda_OS+eye(size(Pnew,1))+Hnew' * M_OS * Hnew)^(-1));% update gain matrix OS
% variance matrix
M_AE = (1/lamda_AE)* (M_AE -( K_AE * Hnew' * M_AE));% updating variance matrix  AE 
M_OS = (1/lamda_OS)*(M_OS -( K_OS * Hnew' * M_OS));% updating variance matrix  AE 
% Output weights
B_AE = B_AE + M_AE * Hnew * (E{t});% update B_OS
B_OS= B_OS+ M_OS*Hnew*(En{t});
end % end USS
lamdas_AE(t,1)=lamda_AE;
lamdas_OS(t,1)=lamda_OS;
end
end
end_time_train=cputime;
TrainingTime=end_time_train-start_time_train;
%%%%%% training accuracy
for t=1:numel(Trinputs)% coding
   input=Trinputs{t};
   H_t=pinv(B_AE')*input';
   %%%%%Activation function%%%
 switch lower(activF)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_t = 1 ./ (1 + exp(-H_t));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_t = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_t = double(hardlim(H_t));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_t = tribas(H_t);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_t = radbas(H_t);
    case {'relu'}
        %%%%%%%% linear function
        H_t = max(H_t,0);
        %%%%%%%% More activation functions can be added here                
end
   Troutputs_hat{t}= H_t'*B_OS;
   Troutputs_hat{t}=scaledata(Troutputs_hat{t},min(Troutputs{t}),max(Troutputs{t}));
   Tr_acc(t)  = sqrt(mse(Troutputs_hat{t}-Troutputs{t}));
end
 trainingAccuracy=mean(Tr_acc);
 %%%%%% testing accuracy
 start_time_test=cputime; 
 for t=1:numel(Tsinputs)
  input=Tsinputs{t};
  H_t=pinv(B_AE')*input';
   %%%%%Activation function%%%
 switch lower(activF)
    case {'sig','sigmoid'}
        %%%%%%%% Sigmoid 
        H_t = 1 ./ (1 + exp(-H_t));
    case {'sin','sine'}
        %%%%%%%% Sine
        H_t = sin(tempH);    
    case {'hardlim'}
        %%%%%%%% Hard Limit
        H_t = double(hardlim(H_t));
    case {'tribas'}
        %%%%%%%% Triangular basis function
        H_t = tribas(H_t);
    case {'radbas'}
        %%%%%%%% Radial basis function
        H_t = radbas(H_t);
    case {'relu'}
        %%%%%%%% linear function
        H_t = max(H_t,0);
        %%%%%%%% More activation functions can be added here                
end
  Tsoutputs_hat{t}=(H_t'*B_OS);
  Tsoutputs_hat{t}=scaledata(Tsoutputs_hat{t},min(Tsoutputs{t}),max(Tsoutputs{t}));
  Ts_acc(t)  = sqrt(mse(Tsoutputs_hat{t}-Tsoutputs{t}));
 end
 end_time_test=cputime;
 TestingTime=end_time_test-start_time_test;
 testingAccuracy=mean(Ts_acc);
 %

 %%%% save the training model%%%% 
 net.Tr_Time=TrainingTime;
 net.Ts_Time=TestingTime;
 net.Tr_RMSE=trainingAccuracy;
 net.Ts_RMSE=testingAccuracy;
 net.lamdas_AE=lamdas_AE;
 net.lamdas_OS=lamdas_OS;
 net.Tsoutputs_hat=Tsoutputs_hat;
 net.e_AE=e_AE;
 net.e_OS=e_OS;
 net.Index=Index;
 net.Cr=Cr;

 
end
