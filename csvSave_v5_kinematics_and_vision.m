% Matching vision and kinematics robust to Data collection Protocol Errors. With UIDs
% This version uses the new 'getMatchingVisionIndices.m' file in the 'utilfunctions' directory. That file also uses load_pupil_timestamps.m files which loads pupil timestamps from .csv file in the 
% pupil's exports folder.

% - Creates "missing_xsens_timestamps.txt" file in the "subject#/unstructured/trial#" folder. The txt files contains the # of missing timestamps and a list of 
%   pupil indices which don't have a corresponding xsens timestamp. Following are the cases that might have caused missing timestamps:
%   1) If the first index listed is a large # and all other indices are consecutive (e.g. 9000,9001,9002,9003.... and so on), then xsens recording was stopped 
%      before pupil. 
%      SOLUTION: Drop the pupil images corresponding to the listed indices and load rest as inputs in this case.  

%   2) If the indices listed are 1,2,3,4,5,6,7....... and so on, then pupil recording was started before xsens. 
%      SOLUTION: In this case again drop the frames corresponding to listed indices and load the rest as inputs

%   3) If the text file says "All indices missing" then the pupil and the xsens session naming has been messed up and the two don't correspond to each other. 
%      SOLUTION: Check if the recording time of both pupil and xsens to confirm and rename correctly.

%   4) If text file has randomly varying list of indices, then xsens might have also dropped some frames. This would rarely be the case.
%      SOLUTION: Don't use this trial.

% - The python file for extracting jpgs from the pupil videos, also creates "frames_dropped.txt" file in the "subject#/unstructured/trial#" folder. 
%   This happens if FFMPEG drops unidentified frames due to some decoding/encdoing problem. This is an issue which should happen rarely. 
%   SOLUTION: Don't use the trial.

%Lists vision frames extracted from ffmpeg and saves in csv column under 'frames'



%%
clear all; clc;close all;
addpath('F:\Vision_Data','F:\Vision_Data\matlab_code','F:\Vision_Data\xSens_Phase2','F:\Unstructured_data\Unstructured_Data')
cd('F:\Vision_Data')
this_dir = pwd;

idcs   = strfind(this_dir,filesep);
main_dir = this_dir(1:idcs(end)-1); %main data dir
data_dir='Unstructured_data\Unstructured_Data';
subjects = {'xUD002'}



raw_mvnx_folder='raw_mvnx'
pupil_data_folder='pupil'
%possile activity list in this dataset
activities_list={'test','obstacle','unstructured'}

%%

num_subjects = size(subjects,2)
bInit=true;idx_file=0;initialVars={};skipped_files = {}
for idx_subject = 1:1:num_subjects
    
    subdirectory_name = fullfile(main_dir,data_dir,  subjects{1,idx_subject})
    this_subject_mvnx_dir=fullfile(subdirectory_name,raw_mvnx_folder)
    this_subject_pupil_dir=fullfile(subdirectory_name,pupil_data_folder)
    
    listing=dir(this_subject_mvnx_dir);
    
    %remove all non mvnx files
    listing=listing(~ismember({listing.name},{'.','..','junk','desktop.ini','skipped_files'}));
    
    num_files=size(listing,1);
    %initialize this subject position(angle), angular velocity and
    %acceleration mats
    
    this_sub_all_trials_angPos=[];this_sub_all_trials_angVel=[];this_sub_all_trials_angAcc=[]; this_sub_all_trials_time=[];
    if bInit
        disp('Init')
        initialVars = who
        bInit=false;
    end
    
    
    
    for idx_file = 1%:1:num_files
        initialVars;
        % init clean data mat for every new user
        clearvars  ( '-except',initialVars{:} )
        
        this_filename=listing(idx_file).name;
        current_full_file=fullfile(this_subject_mvnx_dir,this_filename)
        
        dot_idx=strfind(this_filename,'.') -1;
        this_trialName=this_filename(1:dot_idx);
        this_trial_pupilFolder=fullfile(this_subject_pupil_dir,this_trialName)
        % get Activity name
        [this_activityName,idx_activity,this_trial_num] = getActivityName(this_filename,activities_list)
        
        if idx_activity == 0
            % activity match not found; add file to skipped files list and
            % continue to next file
            skipped_files{end+1}=this_filename;
            disp('skipping this file')
            continue;
        end
        
        
        % if no error, load mvnx file
        try
            tree = load_mvnx(current_full_file);
            
        catch ME
            % error loading
            disp(' skipping to next file ')
            skipped_files{end+1}= this_filename;
            continue;
        end
        
        %%
        
        % read some basic data from the file
        
        fileComments = tree.subject.comment;
        frameRate = tree.subject.frameRate;
        suitLabel = tree.subject.label;
        originalFilename = tree.subject.originalFilename;
        recDate = tree.subject.recDate;
        nJoints=size((tree.subject.joints.joint),2); % num of joints
        jointLabels=struct('label', {tree.subject.joints.joint(1:nJoints).label}); % joint labels
        
        
        %retrieve the data frames from the subject
        nSamples = length(tree.subject.frames.frame);
        idx_Start =1; % number of samples to skip; 1 if nothing to be sikipped
        
        normalCount=0;
        %pre allocate some memory for the position of Segment1
        %read the data from the structure e.g. segment 1
        idxNormalSamples=[]; angleDataRaw=[];headers_ang={};allRawData=[];headers_time_ang_vel_acc={};
        jAngleMinMaxScalers=[];angleDataNorm=[];
        jVelMinMaxScalers=[];velDataNorm=[];
        jAccMinMaxScalers=[];accDataNorm=[];
        norm_range=[0,1]; anatomicalPlanes={'Frontal','Trans','Sagittal'};
        
        for iter_Samples=[idx_Start:nSamples]
            if strcmp(tree.subject.frames.frame(iter_Samples).type,'normal') %ignore calibration poses etc, only "normal" ones
                normalCount = normalCount + 1;
                idxNormalSamples=[idxNormalSamples;iter_Samples]; % 4: end
            end % if normal
        end %iter_Samples
        
        %% get matching indices and pupil timestamps
[matchedIndices,pupTimeStamps,matchedIndicesPupil,missingIndicesPupil] = getMatchingVisionIndices(this_trial_pupilFolder,cell2mat({tree.subject.frames.frame(idxNormalSamples).ms}));
     
     if length(matchedIndices) ~= length(pupTimeStamps)
       fid = fopen(fullfile(subdirectory_name, this_activityName, this_trial_num,'missing_xsens_timestamps.txt'),'w');
       if isempty(matchedIndices)
       fprintf(fid,'%s','All indices missing');
       fclose(fid);
       break
       else
       fprintf(fid,'%s\n',strcat(string(length(pupTimeStamps)-length(matchedIndices)),' indices missing \r\n'));
       fprintf(fid,'missing indices for pupil frame #: \r\n');
       fprintf(fid,'%d\r\n',missingIndicesPupil);
       fclose(fid);
       strcat(string(length(pupTimeStamps)-length(matchedIndices)),' indices missing')
       end
     end    
     % idxNormal or not needed?
     matchedNormalIdx=idxNormalSamples(matchedIndices);
%      matchedNormalIdx=matchedIndices;   
        %%
        %create new struct for keeping data from only the matched indices
        jointsData_TempStruct=struct('relative_time',{tree.subject.frames.frame(matchedNormalIdx).time},'time',{tree.subject.frames.frame(matchedNormalIdx).ms},'jointAngle',{tree.subject.frames.frame(matchedNormalIdx).jointAngle},'centerOfMass',{tree.subject.frames.frame(matchedNormalIdx).centerOfMass},'index',{tree.subject.frames.frame(matchedNormalIdx).index});
        
        % create empty structs to populate angles velocities and accelerations 
        jAngles=struct();  jVelocities=struct(); jAccelerations=struct(); norm_range=[0,1];
        centerOfMass=struct();
        jAnglesNorm=struct();jVelocitiesNorm=struct();jAccelerationsNorm=struct();  jAngleMinMaxScaler=struct();
        
        % populate joint angles data for all joints and planes
        for idxMatched=1:numel(matchedNormalIdx)
                    comDataRaw(idxMatched,:)=jointsData_TempStruct(idxMatched).centerOfMass(1,:);                                                            
                    angleDataRaw(idxMatched,:)=jointsData_TempStruct(idxMatched).jointAngle(1,:);
        end% idx
       
                    deltaCOMRaw = diff(comDataRaw);
                    deltaCOMRaw = [deltaCOMRaw;deltaCOMRaw(end,:)];
 %%     
        index=[jointsData_TempStruct.index];
        absTime=[jointsData_TempStruct.time];
        % timestamps in Unix format
        absTimeSec=absTime/1000; ismilliSec=false;
        relTime=[jointsData_TempStruct.relative_time];
        
        
        uid=strcat(subjects{1,idx_subject},"/",this_activityName,"/", this_trial_num)
        uid_col=num2cell(repmat(uid,size(comDataRaw,1),1));
        
        %% for every joint angle, generate velocity, accelrations and normalized data
   
            for idxJoints=1:nJoints
                for idxPlanes=1:3 % 3 anatomic planes
                    
                    
                    
                     %1. get Raw angle velocities and accelarations
                [ velDataRaw(:,(3*(idxJoints-1)+idxPlanes)), accDataRaw(:,(3*(idxJoints-1)+idxPlanes))]= getVelocityAndAcceleration(angleDataRaw(:,(3*(idxJoints-1)+idxPlanes)),absTimeSec,ismilliSec)         ;

                
                %2. Normalize all Raw angles, velocitues and accelerations
              
                
                [angleDataNorm(:,(3*(idxJoints-1)+idxPlanes)),min_val,max_val]= norm_minmax(angleDataRaw(:,(3*(idxJoints-1)+idxPlanes)),norm_range);
                jAngleMinMaxScalers=[jAngleMinMaxScalers,[min_val;max_val]];
                
                [velDataNorm(:,(3*(idxJoints-1)+idxPlanes)),min_valVel,max_valVel]= norm_minmax(velDataRaw(:,(3*(idxJoints-1)+idxPlanes)),norm_range);
                jVelMinMaxScalers=[jVelMinMaxScalers,[min_valVel;max_valVel]];
                
                [accDataNorm(:,(3*(idxJoints-1)+idxPlanes)),min_valAcc,max_valAcc]= norm_minmax(accDataRaw(:,(3*(idxJoints-1)+idxPlanes)),norm_range);
                jAccMinMaxScalers=[jAccMinMaxScalers,[min_valAcc;max_valAcc]];

                    
              
                    
                    
                    
                end %planes
            end %joints
%% List names of vision frames
        frames_dir = fullfile(subdirectory_name, this_activityName, this_trial_num, 'frames');
        cd(frames_dir)
        frame_list = dir('op*.*');
        frame_list = struct2cell(frame_list);
        frame_list_names = frame_list(1,:);
    %% generate header names
        for idxJoints=1:nJoints
            for idxPlanes=1:3 % 3 anatomic planes
                headers_ang{3*(idxJoints-1)+idxPlanes}=strcat('ang','_',jointLabels(idxJoints).label,'_', anatomicalPlanes{idxPlanes})  ;
                headers_vel{3*(idxJoints-1)+idxPlanes}=strcat('vel','_',jointLabels(idxJoints).label,'_', anatomicalPlanes{idxPlanes})  ;
                headers_acc{3*(idxJoints-1)+idxPlanes}=strcat('acc','_',jointLabels(idxJoints).label,'_', anatomicalPlanes{idxPlanes})  ;

            end %planes
        end %joints
        
            % 3 COM directions
                headers_com{1}=strcat('COM','_','x')  ;
                headers_com{2}=strcat('COM','_','y')  ;
                headers_com{3}=strcat('COM','_','z')  ;
            %COM_XYZ
        
        allTimeData(:,1)=relTime; headers_time_vis_com_ang_vel_acc{1}='relTime';
        allTimeData(:,2)=absTimeSec ;headers_time_vis_com_ang_vel_acc{2}='absTime';
        allTimeData = num2cell(allTimeData);
        allRawData = [num2cell(angleDataRaw) num2cell(velDataRaw) num2cell(accDataRaw) num2cell(comDataRaw)];
        allNormData =[num2cell(angleDataNorm) num2cell(velDataNorm) num2cell(accDataNorm) num2cell(deltaCOMRaw)];
        
 %% Remove empty cells corresponding to missing vision or kinematics (NEEDS UPDATE: DEPENDING ON MISSING KINEMATICS INDICES FOR PUPIL INDICES)
 %% WOULD WORK IN MOST LIKELY CASES... FOR NOW JUST REMOVE SESSIONS WITH "frames dropped" or "missing indices" FLAGS
        if length(allRawData)<length(frame_list_names)
%             allRawData = [allRawData ; cell(length(frame_list_names)-length(allRawData),size(allRawData,2))];
%             allNormData = [allNormData ; cell(length(frame_list_names)-length(allNormData),size(allNormData,2))];
%             allTimeData = [allTimeData ; cell(length(frame_list_names)-length(allTimeData),size(allTimeData,2))];
              frame_list_names = frame_list_names(1:length(allRawData));
        elseif length(allRawData)>length(frame_list_names)
%             frame_list_names = [frame_list_names ; cell(length(allTimeData)-length(frame_list_names),1)];
              allRawData = allRawData(1:length(frame_list_names),:);
              allNormData = allNormData(1:length(frame_list_names),:);
              allTimeData = allTimeData(1:length(frame_list_names),:); 
        end        
        
        allTimeData(:,3)=frame_list_names; headers_time_vis_com_ang_vel_acc{3}='frames'; headers_time_vis_com_ang_vel_acc{4}='uid';
        
        allRawData=[allTimeData uid_col allRawData]; allNormData=[allTimeData uid_col allNormData]; headers_time_vis_com_ang_vel_acc=horzcat( headers_time_vis_com_ang_vel_acc,headers_ang,headers_vel,headers_acc,headers_com);
        
        % ^^^ FRAME NAMES ARE IN allTIMEData array 
        
        
        %% bool to save  files
        
        bSave=true;
        if bSave ==true
            
            
            rawFolderName= fullfile(subdirectory_name, this_activityName, this_trial_num, 'raw');  mkdir(rawFolderName);
            normFolderName=fullfile(subdirectory_name, this_activityName, this_trial_num, 'norm'); mkdir(normFolderName);
            
            rawFileName=fullfile(rawFolderName,'jointDataRaw.csv'); normFileName=fullfile(normFolderName,'jointDataNorm.csv');
            angleScalerFileName=fullfile(normFolderName,'jAngleMinMaxScaler.csv');
            velScalerFileName=fullfile(normFolderName,'jVelMinMaxScaler.csv');           
            accScalerFileName=fullfile(normFolderName,'jAccMinMaxScaler.csv');
            
            csvwrite_with_headers_FRAMES(rawFileName,allRawData,headers_time_vis_com_ang_vel_acc);
            csvwrite_with_headers_FRAMES(normFileName,allNormData,headers_time_vis_com_ang_vel_acc);
            csvwrite_with_headers(angleScalerFileName,jAngleMinMaxScalers,headers_ang); %% maybe combine all scalers
            csvwrite_with_headers(velScalerFileName,jVelMinMaxScalers,headers_vel);
            csvwrite_with_headers(accScalerFileName,jAccMinMaxScalers,headers_acc);


            
            
            
        end %bSave
        
        
        
    end %num_files
    
    % write this subjects skipped files
    csvwrite(fullfile(this_subject_mvnx_dir,'skipped_files'),skipped_files)
    
end %subjects

