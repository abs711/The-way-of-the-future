function [angPos,angVel] = getAngularVelocity(angPos,time_stamps,isUnixTime)
% version that calculates speed in degs/sec since xSens timestamps unix
% timestamps 
t=time_stamps;
angVel=zeros(size(angPos));

%if timestamps in unix, multiply by 1000
if isUnixTime
    for i = 1:length(angPos)-1
        angVel(i) = (angPos(i+1)-angPos(i))*1000/(t(i+1)-t(i)) ;
    end
    
else
    for i = 1:length(angPos)-1
        angVel(i) = (angPos(i+1)-angPos(i))/(t(i+1)-t(i)) ;
    end
    
end

angPos=angPos(1:end-1);
angVel=angVel(1:end-1);
end

