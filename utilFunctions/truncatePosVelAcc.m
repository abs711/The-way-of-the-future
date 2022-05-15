function [angPos_trunc,angVel_trunc,angAcc] =  truncatePosVelAcc(angPos,angVel,angAcc)

minNumElements=numel(angAcc);

angPos_trunc=angPos(1:minNumElements);
angVel_trunc=angVel(1:minNumElements);

end
