
%
%v 5
% csv save with headers for kinematics only


%%
clear; clc;close all;
this_dir = pwd;

idcs   = strfind(this_dir,filesep);
main_dir = this_dir(1:idcs(end-1)-1); %main data dir
raw_mvnx_folder='raw_mvnx_expt'

data_dir='xSens_Phase1';
%subjects = {'xMF03'}%,'xMF04','xMF05','xMF06','xMF08','xMF09','xMF10','xMF11','xMF12','xMF13'}
%subjects = {'xMFs009','xMFs011','xMFs012','xMFs013','xMFs015','xMFs016','xMFs019','xMFs024','xMFs025','xMFs026','xMFs030'}
%subjects = {'xMFs068','xMFs069','xMFs073','xMFs076','xMFs079','xMFs080','xMFs081','xMFs082','xMFs085','xMFs087','xMFs090'};

%activities_list={'test','flat','stop','stair'} %possile activity list in this dataset

subjects = {'xOA01','xOA07','xOA24'}
activities_list={'Obstacle'}
% data_dir='xSens_Phase2';
%subjects = {'xOA01','xOA02','xOA03','xOA04','xOA05','xOA06','xOA07','xOA08','xOA09','xOA10','xOA22','xOA23','xOA24'}
%subjects = {'xOA22','xOA23','xOA24'}
%activities_list={'Edgren','Illinois','Test','Obstacle'} %possile activity list in this dataset



%%

num_subjects = size(subjects,2)
bInit=true;idx_file=0;initialVars={};skipped_files = {}
for idx_subject = 1:1:num_subjects
    
    subdirectory_name = fullfile(main_dir,data_dir,'raw',  subjects{1,idx_subject})
    processed_dir= fullfile(main_dir,data_dir,'processed', subjects{1,idx_subject})
    this_subject_mvnx_dir=fullfile(subdirectory_name,raw_mvnx_folder)
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
    
    
    
    for idx_file =1:1:num_files
        initialVars;
        % init clean data mat for every new user
        clearvars  ( '-except',initialVars{:} )
        
        this_filename=listing(idx_file).name;
        current_full_file=fullfile(this_subject_mvnx_dir,this_filename)
        
        
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
            %load('tree.mat') % to test, loads quickly
        catch ME
            % error loading
            disp(' skipping to next file ')
            skipped_files{end+1}= this_filename;
            continue;
        end
        
        %%
        
        % read some basic data from the file
        
        %fileComments = tree.subject.comment;
        frameRate = tree.subject.frameRate;
        %suitLabel = tree.subject.label;
        originalFilename = tree.subject.originalFilename;
        recDate = tree.subject.recDate;
        nJoints=size((tree.subject.joints.joint),2); % num of joints
        jointLabels=struct('label', {tree.subject.joints.joint(1:nJoints).label}); % joint labels
        
        
        %retrieve the data frames from the subject
        nSamples = length(tree.subject.frames.frame)
        idx_Start =1; % number of samples to skip; 1 if nothing to be sikipped
        
        normalCount=0;
        %pre allocate
        idxNormalSamples=[]; angleDataRaw=[];headers_ang={};allRawData=[];headers_time_ang_vel_acc={};
        jAngleMinMaxScalers=[];angleDataNorm=[];
        jVelMinMaxScalers=[];velDataNorm=[];
        jAccMinMaxScalers=[];accDataNorm=[];
        norm_range=[0,1]; anatomicalPlanes={'Frontal','Trans','Sagittal'};
        
        for iter_Samples=[idx_Start:nSamples]
            if strcmp(tree.subject.frames.frame(iter_Samples).type,'normal') %ignore calibration poses etc, only "normal" ones
                normalCount = normalCount + 1;
                idxNormalSamples=[idxNormalSamples;iter_Samples];
            end % if normal
        end %iter_Samples
        
        %%
        %create new struct for keeping data
        
        jointsData_struct=struct('relative_time',{tree.subject.frames.frame(idxNormalSamples).time},'time',{tree.subject.frames.frame(idxNormalSamples).ms},'jointAngle',{tree.subject.frames.frame(idxNormalSamples).jointAngle});
        %jointsData_struct=struct('relative_time',{tree.subject.frames.frame(idxNormalSamples).time},'jointAngle',{tree.subject.frames.frame(idxNormalSamples).jointAngle}); % xMF02 has no bas time
        

        relTime=[jointsData_struct.relative_time]';
      
        absTime=[jointsData_struct.time]'; 
        %absTime=1520425096  + relTime; % approx abs time : xMF02 has no bas time
        
        % timestamps in Unix format
        absTimeSec=absTime/1000; ismilliSec=false;
        
        
        
        
        % populate joint angles data for all joints and planes
        for idxNormal=1:normalCount
            angleDataRaw(idxNormal,:)=jointsData_struct(idxNormal).jointAngle(1,:);
        end% idxfor idxNormal=1:normalCount
        
        
        % timestamps in Unix format
        isUnixTime=true
        
        uid=strcat(subjects{1,idx_subject},"/",this_activityName,"/", this_trial_num)
        uid_col=repmat(uid,normalCount,1);
        %  %2. Normalize all Raw angles, velocitues and accelerations
        for idxJoints=1:nJoints
            for idxPlanes=1:3 %3planes
                
                
                %1. get Raw angle velocities and accelarations
                [ velDataRaw(:,(3*(idxJoints-1)+idxPlanes)), accDataRaw(:,(3*(idxJoints-1)+idxPlanes))]= getVelocityAndAcceleration(angleDataRaw(:,(3*(idxJoints-1)+idxPlanes)),absTimeSec,ismilliSec)         ;
                
                
                %2. Normalize all Raw angles, velocitues and accelerations
                
                
                [angleDataNorm(:,(3*(idxJoints-1)+idxPlanes)),min_val,max_val]= norm_minmax(angleDataRaw(:,(3*(idxJoints-1)+idxPlanes)),norm_range);
                jAngleMinMaxScalers=[jAngleMinMaxScalers,[min_val;max_val]];
                
                [velDataNorm(:,(3*(idxJoints-1)+idxPlanes)),min_valVel,max_valVel]= norm_minmax(velDataRaw(:,(3*(idxJoints-1)+idxPlanes)),norm_range);
                jVelMinMaxScalers=[jVelMinMaxScalers,[min_valVel;max_valVel]];
                
                [accDataNorm(:,(3*(idxJoints-1)+idxPlanes)),min_valAcc,max_valAcc]= norm_minmax(accDataRaw(:,(3*(idxJoints-1)+idxPlanes)),norm_range);
                jAccMinMaxScalers=[jAccMinMaxScalers,[min_valAcc;max_valAcc]];
                
            end % planes
        end %joints
        
        
        
        
        
        %% generate header names
        for idxJoints=1:nJoints
            for idxPlanes=1:3 % 3 anatomic planes
                headers_ang{3*(idxJoints-1)+idxPlanes}=strcat('ang','_',jointLabels(idxJoints).label,'_', anatomicalPlanes{idxPlanes})  ;
                headers_vel{3*(idxJoints-1)+idxPlanes}=strcat('vel','_',jointLabels(idxJoints).label,'_', anatomicalPlanes{idxPlanes})  ;
                headers_acc{3*(idxJoints-1)+idxPlanes}=strcat('acc','_',jointLabels(idxJoints).label,'_', anatomicalPlanes{idxPlanes})  ;
                
            end %planes
        end %joints
        
        headers_time_ang_vel_acc{1}='relTime';
        headers_time_ang_vel_acc{2}='absTime';
        headers_time_ang_vel_acc{3}='uid';
        headers_ang_vel_acc=horzcat(headers_ang,headers_vel,headers_acc);
        headers_time_ang_vel_acc=horzcat( headers_time_ang_vel_acc,headers_ang,headers_vel,headers_acc);
        
        allRawData=[relTime absTime uid_col angleDataRaw velDataRaw accDataRaw];
        allNormData=[relTime absTime uid_col angleDataNorm velDataNorm accDataNorm];
        
        
        jMinMaxScalers= [jAngleMinMaxScalers jVelMinMaxScalers jAccMinMaxScalers];
        
        %% bool to save  files
        
        bSave=true;
        if bSave ==true
            
            
            rawFolderName= fullfile(processed_dir, this_activityName, this_trial_num, 'raw');  mkdir(rawFolderName);
            normFolderName=fullfile(processed_dir, this_activityName, this_trial_num, 'norm'); mkdir(normFolderName);
            
            rawFileName=fullfile(rawFolderName,'jointDataRaw.csv'); normFileName=fullfile(normFolderName,'jointDataNorm.csv')
            
            jMinMaxScalerFileName=fullfile(normFolderName,'jMinMaxScaler.csv');
            
            writeDataWithHeaders(rawFileName,allRawData,headers_time_ang_vel_acc)
            writeDataWithHeaders(normFileName,allNormData,headers_time_ang_vel_acc);
            csvwrite_with_headers(jMinMaxScalerFileName,jMinMaxScalers,headers_ang_vel_acc); 
            

            
            
        end %bSave
        
        
        
    end %num_files
    
    % write this subjects skipped files
    csvwrite(fullfile(this_subject_mvnx_dir,'skipped_files'),skipped_files)
    
end %subjects

function writeDataWithHeaders(filename,data_mat,headers)
 writetable(array2table(data_mat,"VariableNames",headers),filename)
end
