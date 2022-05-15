function [angVel,angAcc] = getVelocityAndAcceleration(angPos,t,ismilliSec)



angVel=zeros(size(angPos));
angAcc=zeros(size(angPos));
if ismilliSec 

%if timestamps in unix, multiply by 1000

    for i = 1:length(angPos)-1
        angVel(i) = (angPos(i+1)-angPos(i))*1000/(t(i+1)-t(i)) ;
    end
    

   for i = 1:length(angPos)-1
        angAcc(i) = (angVel(i+1)-angVel(i))*1000/(t(i+1)-t(i)) ;
   end


end


if ~ismilliSec

    for i = 1:length(angPos)-1
        angVel(i) = (angPos(i+1)-angPos(i))/(t(i+1)-t(i)) ;
    end
   

   for i = 1:length(angVel)-1
        angAcc(i) = (angVel(i+1)-angVel(i))/(t(i+1)-t(i)) ;
   end


end

end

