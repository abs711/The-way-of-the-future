function [gait_inits, input_cycles,knee_cycles,knee_cycles_mat,dtw_knee_mat,clusters, medoids, frames_cycles] = get_cycles_clusters(src_dir,filename);

load('input_output_fields.mat');

 % jointDataRaw.frames=get_frame_nums(jointDataRaw.frames);
    
    jointDataRaw = readtable(fullfile(src_dir,filename));
    
    % jointDataRaw.frames=get_frame_nums(jointDataRaw.frames);
    
    contra_knee_sag=jointDataRaw.ang_jLeftKnee_Sagittal;
    num_samples=numel(contra_knee_sag);
    ang_ankle_sag=jointDataRaw.ang_jRightAnkle_Sagittal;
    ang_knee_sag=jointDataRaw.ang_jRightKnee_Sagittal;
    
    %%
    
    [pks_n,locs_n,w,p_n]=findpeaks(contra_knee_sag,'MinPeakDistance',20,'MinPeakHeight',20);
    
    gait_inits=locs_n(find(p_n>15));
    remove_gait_idx=find(gait_inits<200 | gait_inits>(num_samples-400));
    gait_inits(remove_gait_idx(:,1),:)=[];
    
    %%
    %     num_row=size(jointDataRaw,1);
    
    input_jointData=jointDataRaw{:,input_fields(5:end,1)};
    target_jointData=jointDataRaw{:,target_fields(5:end,1)}; %knee_idx=5, ankle_idx=6
    frames_jointData=[jointDataRaw.relTime get_frame_nums(jointDataRaw.frames)];
    % %%
    for i=[1:size(gait_inits,1)-1]
        
        input_cycles{i,1}=input_jointData(gait_inits(i):gait_inits(i+1)-1,:)';
        knee_cycles{i,1}=target_jointData(gait_inits(i):gait_inits(i+1)-1,1)';
        frames_cycles{i,1}=frames_jointData(gait_inits(i):gait_inits(i+1)-1,:)';
    end
    %%
    
    % dtw_mat_contra=zeros(size(knee_cycles,1));
    knee_cycles_mat=resample_cell_to_mat(knee_cycles,100);
    dtw_knee_mat=squareform(pdist(knee_cycles_mat,@dtw_mat_dist));
    
    %%
    k=3;
     [clusters,medoids,sumd,D,midx,info]=kmedoids(knee_cycles_mat,k,'Distance',@dtw_mat_dist,'Start','cluster','Options',statset('MaxIter',100));
%   [clusters,medoids,sumd,D,midx,info]=kmedoids(knee_cycles_mat,k,'Distance',@dtw_mat_dist,'Start','cluster');

end 