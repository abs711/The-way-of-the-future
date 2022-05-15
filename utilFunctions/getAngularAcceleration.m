function [angVel,angAcc] = getAngularAcceleration(angVel,t,isUnixTime)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

angAcc=zeros(size(angVel));

%if timestamps in unix, multiply by 1000
if isUnixTime
    for i = 1:length(angVel)-1
        angAcc(i) = (angVel(i+1)-angVel(i))*1000/(t(i+1)-t(i)) ;
    end
    
else
    for i = 1:length(angVel)-1
        angAcc(i) = (angVel(i+1)-angVel(i))/(t(i+1)-t(i)) ;
    end
    
end

angVel=angVel(1:end-1);
angAcc=angAcc(1:end-1);
end

