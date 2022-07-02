%data prep for LSTM
% desired X gait_Steps x input features, Y gait steps by outputs

clc; close all; clear all
data_dir = "D:/GDrive_UW/DeepLearning/data/xSens_Phase2/" ;

 subjects={'xOA02'}
% trials={'005','003','004'}

%  subjects={'xUD002','xUD004','xUD006','xUD006','xUD007','xUD008','xUD009','xUD011','xUD012','xUD015'};

for sub_idx=1:numel(subjects)
    dirs_list=dir(fullfile(data_dir,subjects{sub_idx},'Test'))
    dirs_list=dirs_list(~ismember({dirs_list.name},{'.','..','junk','desktop.ini','skipped_files'}));
    trials=extractfield(dirs_list,'name')

for iter_trial=1:numel(trials)
    src_dir=  fullfile(data_dir,subjects{sub_idx},'Test',trials{iter_trial});
    filename = "raw/jointDataRaw.csv";
    dest_dir=fullfile(data_dir,subjects{sub_idx},'Test',trials{iter_trial})
    
   [gait_inits, input_joints_cycles,knee_cycles,knee_cycles_mat,matlab_dtw_dist ,matlab_intrasub_clusters, matlab_medoids]= get_cycles_clusters_noFrames(src_dir,filename);
   
    %%
    saveOut=0
    if saveOut

        if exist(dest_dir, 'dir')
%             warning(...
        else
            mkdir(dest_dir)
        end
        output_mat=fullfile(dest_dir,'normalizedT_gait_cycles_data.mat')
        save (output_mat, 'gait_inits', 'input_joints_cycles' ,'knee_cycles','knee_cycles_mat','matlab_dtw_dist' ,'matlab_intrasub_clusters', 'matlab_medoids', 'frames_cycles')
    end
    
end %trials

end %subs


