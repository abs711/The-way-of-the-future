function [input_Indices,output_Indices] = getInputandTargetIndex(outputJoint,outputType)
%GETINPUTANDTARGETINDEX Summary of this function goes here
%   Detailed explanation goes here


if (outputType == 'angPos')
    
    %   automate this by output name
    %     idx_jL5S1 =[1:3];
    %     idx_jL4L3 =[4:6];
    %     idx_jL1T12=[7:9];
    %     idx_jT9T8 =[10:12];
    %     idx_jT1C7=[13:15];
    %     idx_jC1Head =[16:18];
    %     idx_jRightC7Shoulder=[19:21];
    %     idx_jRightShoulder= [ 22:24];
    %     idx_jRightElbow = [25:27];
    %     idx_jRightWrist=[28:30];
    %     idx_jLeftC7Shoulder=[31:33];
    %     idx_jLeftShoulder=[34:36];
    %     idx_jLeftElbow = [37:39];
    %     idx_jLeftWrist = [40:42];
    %     idx_jRightHip=[43:45];
    %     idx_jRightKnee=[46:48];
    %     idx_jRightAnkle = [49:51];
    %     idx_jRightBallFoot = [52:54];
    %     idx_jLeftHip=[55:57];
    %     idx_jLeftKnee=[58:60];
    %     idx_jLeftAnkle=[61:63];
    %     idx_jLeftBallFoot=[64:66];
    
    if (outputJoint == 'LeftAnkle')
        input_Indices=[1:60,64:66];
        output_Indices=[61:63];
    elseif (outputJoint == 'RightAnkle')
        input_Indices=[1:48,52:66];
        output_Indices=[49:51];
    end
    
    
elseif outputType =='angVel'
    
    
    
end


end

