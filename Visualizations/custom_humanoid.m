clear all; close all;
tree = load_mvnx('.\xUD004\UnstructuredH\UnstructuredC-0011'); %Path to kinematics data
im_path = '.\xUD004\UnstructuredC\0011\frames'; %Path to image frames
time_stamp_file = '.\xUD004\UnstructuredC\0011\jointDataRaw_ss6.csv'; %Get the timestamps to visualize. SS6 refers to frames subsampled by a factor of 6
jointDataRawss6 =  importTimeStamps(time_stamp_file);
timestamps = jointDataRawss6.absTime*1000;
frame_list = jointDataRawss6.frames; 

for i = 1:23
segments{i} = tree.subject.segments.segment(i).label; % Get the body segments e.g. upper arm, lower arm etc.
end

for i = 1:22
joint_labels{i} = tree.subject.joints.joint(i).label; % Get the joint labels e.g. elbow, knee etc.
end

npose = 0
rotation_true = 1
camera_placement = [-1 0 0.3]




generate_static_character(npose) % Generate static character in the Xsens neutral pose
%% Get Rotations
flag = 0;
ts_window = [];
for ts = 1:length(timestamps)
% rotation_vectors = tree.subject.frames.frame(frame_num).orientation;
ind = find([tree.subject.frames.frame.ms] == timestamps(ts));

if length(ind) > 0 && flag == 0
    flag = 1;
    first_instance = ts;
end

if length(ind) > 0 && flag == 1
    ts_window = [ts_window ts];
end

if isempty(ind) && flag == 1
    last_instance = ts-1;
    break
end

if flag == 1
    current_rotation_vectors = tree.subject.frames.frame(ind).orientation;
    if ts == first_instance
        rotation_vectors = get_rotations(current_rotation_vectors,rotation_true);
    else
        rotation_vectors = [rotation_vectors get_rotations(current_rotation_vectors,rotation_true)];
    end
end

end



set(gca,'nextplot','replacechildren');
v = VideoWriter('visualization2.avi');
v.FrameRate=5;
open(v);

f = figure(2)
hold on;
f.WindowState = 'maximized';

%% Build
flag = 0;
for ts = 1:length(ts_window)
% profile on    
% rotation_vectors = tree.subject.frames.frame(frame_num).orientation;
ind = find([tree.subject.frames.frame.ms] == timestamps(ts_window(ts)));

if length(ind) > 0
    flag = 1;
    first_instance2 = ts_window(ts);
end

if isempty(ind) && flag == 1
    last_instance2 = ts_window(ts);
    break
end

if flag == 1
character_root = [0 0 0];



      generate_character(npose,segments,character_root,rotation_vectors(:,ts),camera_placement,ts)
      generate_videostream(frame_list,ts_window(ts),im_path)
      frame = getframe(gcf);
      size(frame.cdata)
      writeVideo(v,frame);

end
end

close(v);