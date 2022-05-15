function [matchedIndicesXSens,y] = getMatchingVisionIndices(pupilFolder,xSensTimeStamps_ms)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here



load(fullfile(pupilFolder,'world_timestamps.mat')) ;%% load pupil


imp = importfile(fullfile(pupilFolder,'info.csv')); %% import info file 
ST = imp(4,2); %% Initial Unix timestamp
SYN = imp(5,2); %% Imported pupil timestamps synced wrt this
y = (x + (ST - SYN)*ones(size(x)))*1000; %% Pupil timestamps in ms
%% Selecting frames from Xsens corresponding to Pupil frames
jj  = 1;
tol = 10; %%tolerance in ms
ind = [];
pup = [];
for ii = 1:length(y)
    
   for kk = jj:length(xSensTimeStamps_ms) 
    if (abs(xSensTimeStamps_ms(jj)-y(ii))<tol && abs(xSensTimeStamps_ms(jj+1)-y(ii))>=tol)
        ind = [ind; jj];
        pup = [pup; ii];
        jj = jj+1;
        break
    else   
        jj = jj+1;
    end
   end 
end

matchedIndicesXSens=ind; %% Xsens frame no.
% pup; %% Pupil frame no.




end

