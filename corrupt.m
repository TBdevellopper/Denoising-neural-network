function[Cp]=corrupt(p,N_ratio,dB)
randn ('state',3);
if dB>0
Q=size(p,1);
[noiseInd] = divideint(Q,N_ratio,0,1-N_ratio);

% generate noise
 noise=(randn(length(noiseInd),size(p,2)))*dB;
 p(noiseInd,:)=p(noiseInd,:)+(noise);
%p(noiseInd,:)= p(noiseInd,:)+noise;
 Cp=scaledata(p,0,1);
else
 Cp=p;
end
end