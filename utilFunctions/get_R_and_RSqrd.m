function [pearsons_r,rSquared] = get_R_and_RSqrd(y_Preds,y_Tests)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
num_trials=size(y_Preds,1);
pearsons_r=zeros(1,num_trials);
rSquared=zeros(1,num_trials);
for idx_trials=1:1:num_trials
    
    this_pred=y_Preds(idx_trials,:);
    this_test= y_Tests(idx_trials,:);
    R = corrcoef(this_pred,this_test);
    pearsons_r(idx_trials)=R(1,2);
    
    ssResidual= sum ((this_test- this_pred).^2);
%     ssTotal=  (length(this_test)-1) * var(this_test) 
    ssTotal =sum((this_test- mean(this_test)).^2);
    rSquared(idx_trials) = 1-ssResidual/ssTotal;
    
    
end

end

