function [matchedIndicesXSens,y,matchedIndicesPupil,missingIndicesPupil] = getMatchingVisionIndices(pupilFolder,xSensTimeStamps_ms)
%UNTITLED3 Summary of this function goes here
%   Detailed explanation goes here


x = load_pupil_timestamps(fullfile(pupilFolder,'exports','000','world_timestamps.csv')); %% load pupil

Pupil_timestamps_length = length(x)
xSens_timeStamps_length = length(xSensTimeStamps_ms)

imp = importfile(fullfile(pupilFolder,'info.csv')); %% import info file
ST = imp(4,2); %% Initial Unix timestamp
SYN = imp(5,2); %% Imported pupil timestamps synced wrt this
y = (x + (ST - SYN)*ones(size(x)))*1000; %% Pupil timestamps in ms
%% Selecting frames from Xsens corresponding to Pupil frames
jj  = 1;
tol = 10; %%tolerance in ms
ind = [];
pup = [];
all_pup_inds = 1:length(y); 
for ii = 1:length(y)
    for kk = jj:length(xSensTimeStamps_ms)
        if jj < length(xSensTimeStamps_ms)         
            if (abs(xSensTimeStamps_ms(jj)-y(ii))<tol && abs(xSensTimeStamps_ms(jj+1)-y(ii))>=tol)
                ind = [ind; jj];
                pup = [pup; ii];
                jj = jj+1;
                break
            else
                
                jj = jj+1;
                
            end
            
        else
            break
        end
    end
end
matchedIndicesXSens=ind; %% Xsens frame no.
matchedIndicesPupil=pup; %% Pupil frame no. that got matched
missingIndicesPupil=find(ismember(all_pup_inds,pup)==0); %% Pupil frame no. with a missing Xsens frame
% pup; %% Pupil frame no.

    


end

%%%%%%%%%%%%%%%%%%BETTER??%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ind = find(abs(xSensTimeStamps_ms-y(i))<tol);
% indn = ind(find(abs(xSensTimeStamps_ms(ind+1)-y(i))>=tol))
