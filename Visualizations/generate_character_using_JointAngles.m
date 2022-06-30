function [] = generate_character_using_JointAngles(npose,joint_labels,character_root,rotation_values,camera_placement,idx)
%% Dimensions
hip_width = 26;
hip_height = 93;
knee_height = 48;
Foot_length = 10;
shoulder_height = 146;
shoulder_width = 44;
height = 176;
arm_span = 180;
%% visualization specs
pelvis_specs = 11 ;
thigh_specs = 20 ;
shin_specs = 11 ;
foot_specs = 10 ;
Ball_specs = 8 ;
spine_specs = 15;
shoulder_specs = 8;
T8_specs = 20;
T12_specs = 17; 
biceps_specs = 11;
forearm_specs = 9;
hand_specs = 10;
neck_specs = 11;
head_specs = 15;

segment_increment = 0.5;
neck_head_ratio = 0.3;
arm_ratio = 0.4;% <0.5 only
ball_size = 0.2;
%% Build Vectors

%% Pelvis
pelv_vec = [0,hip_width/2,0];
% pelv_ind = find(strcmp(joint_labels,'Pelvis'));
% pelv_vec = rotate_vec_using_eul(pelv_vec_length,rotation_values(pelv_ind));

%% Upper Legs
Thigh_length = hip_height-knee_height; 

% uLegR_vec_length = Thigh_length;
uLegR_vec = -[0, 0, Thigh_length];

uLegR_ind = find(strcmp(joint_labels,'jRightHip'));
% uLegR_vec = rotate_vec_using_eul(uLegR_vec_length,rotation_values(uLegR_ind,:))
uLegR_vec = rotate_vec_using_eul(uLegR_vec,rotation_values(uLegR_ind,:))

% uLegL_vec_length = Thigh_length;
uLegL_vec = -[0, 0, Thigh_length];

uLegL_ind = find(strcmp(joint_labels,'jLeftHip'));
% uLegL_vec = rotate_vec_using_eul(uLegL_vec_length,rotation_values(uLegL_ind,:))
uLegL_vec = rotate_vec_using_eul(uLegL_vec,rotation_values(uLegL_ind,:))
%% Lower Legs
% lLegR_vec_length = knee_height;

lLegR_vec = -[0, 0, knee_height];

lLegR_ind = find(strcmp(joint_labels,'jRightKnee'));
% lLegR_vec = rotate_vec_using_eul(lLegR_vec_length,rotation_values(lLegR_ind,:))
lLegR_vec = rotate_vec_using_eul(lLegR_vec,rotation_values(lLegR_ind,:))


% lLegL_vec_length= knee_height;
lLegL_vec = -[0, 0, knee_height];

lLegL_ind = find(strcmp(joint_labels,'jLeftKnee'));
% lLegL_vec = rotate_vec_using_eul(lLegL_vec_length,rotation_values(lLegL_ind,:))
lLegL_vec = rotate_vec_using_eul(lLegL_vec,rotation_values(lLegL_ind,:))

%% Feet
% FootR_vec_length= Foot_length;
FootR_vec = [Foot_length, 0, 0];

FootR_ind = find(strcmp(joint_labels,'jRightAnkle'));
% FootR_vec = rotate_vec_using_eul(FootR_vec_length,rotation_values(FootR_ind,:))
FootR_vec = rotate_vec_using_eul(FootR_vec,rotation_values(FootR_ind,:))


% FootL_vec_length= Foot_length;
FootL_vec = [Foot_length,0, 0];

FootL_ind = find(strcmp(joint_labels,'jLeftAnkle'));
% FootL_vec = rotate_vec_using_eul(FootL_vec_length,rotation_values(FootL_ind,:))
FootL_vec = rotate_vec_using_eul(FootL_vec,rotation_values(FootL_ind,:))

%% Balls
% BallR_vec_length= Foot_length*ball_size;
BallR_vec = [Foot_length*ball_size, 0, 0];

BallR_ind = find(strcmp(joint_labels,'jRightBallFoot'));
% BallR_vec = rotate_vec_using_eul(BallR_vec_length,rotation_values(BallR_ind,:))
BallR_vec = rotate_vec_using_eul(BallR_vec,rotation_values(BallR_ind,:))

% BallL_vec_length= Foot_length*ball_size;
BallL_vec = [Foot_length*ball_size, 0, 0];

BallL_ind = find(strcmp(joint_labels,'jLeftBallFoot'));
% BallL_vec = rotate_vec_using_eul(BallL_vec_length,rotation_values(BallL_ind,:))
BallL_vec = rotate_vec_using_eul(BallL_vec,rotation_values(BallL_ind,:))

%% Spine
spine_length = shoulder_height-hip_height;

% L5_vec_length= 0.25*spine_length;
L5_vec = [0 0 0.25*spine_length];

L5_ind = find(strcmp(joint_labels,'jL5S1'));
% L5_vec = rotate_vec_using_eul(L5_vec_length,rotation_values(L5_ind,:))
L5_vec = rotate_vec_using_eul(L5_vec,rotation_values(L5_ind,:))

% L3_vec_length= 0.25*spine_length;
L3_vec = [0 0 0.25*spine_length];

L3_ind = find(strcmp(joint_labels,'jL4L3'));
% L3_vec = rotate_vec_using_eul(L3_vec_length,rotation_values(L3_ind,:))
L3_vec = rotate_vec_using_eul(L3_vec,rotation_values(L3_ind,:))

% T12_vec_length= 0.25*spine_length;
T12_vec = [0 0 0.25*spine_length];

T12_ind = find(strcmp(joint_labels,'jL1T12'));
% T12_vec = rotate_vec_using_eul(T12_vec_length,rotation_values(T12_ind,:))
T12_vec = rotate_vec_using_eul(T12_vec,rotation_values(T12_ind,:))

% T8_vec_length= 0.25*spine_length;
T8_vec = [0 0 0.25*spine_length];

T8_ind = find(strcmp(joint_labels,'jT9T8'));
% T8_vec = rotate_vec_using_eul(T8_vec_length,rotation_values(T8_ind,:))
T8_vec = rotate_vec_using_eul(T8_vec,rotation_values(T8_ind,:))


%% Shoulder-Neck-Head
over_shoulder = height-shoulder_height;

% ShoulderR_vec_length= 0.5*shoulder_width;
ShoulderR_vec = -[0 0.5*shoulder_width 0];

ShoulderR_ind = find(strcmp(joint_labels,'jRightT4Shoulder'));
% ShoulderR_vec = rotate_vec_using_eul(ShoulderR_vec_length,rotation_values(ShoulderR_ind,:))
ShoulderR_vec = rotate_vec_using_eul(ShoulderR_vec,rotation_values(ShoulderR_ind,:))

% ShoulderL_vec_length= 0.5*shoulder_width ;
ShoulderL_vec = [0 0.5*shoulder_width 0];

ShoulderL_ind = find(strcmp(joint_labels,'jLeftT4Shoulder'));
% ShoulderL_vec = rotate_vec_using_eul(ShoulderL_vec_length,rotation_values(ShoulderL_ind,:))
ShoulderL_vec = rotate_vec_using_eul(ShoulderL_vec,rotation_values(ShoulderL_ind,:))

% neck_vec_length= neck_head_ratio*over_shoulder;
neck_vec = [0 0 neck_head_ratio*over_shoulder];

neck_ind = find(strcmp(joint_labels,'jT1C7'));
% neck_vec = rotate_vec_using_eul(neck_vec_length,rotation_values(neck_ind,:))
neck_vec = rotate_vec_using_eul(neck_vec,rotation_values(neck_ind,:))

% head_vec_length= (1-neck_head_ratio)*over_shoulder;
head_vec = [0 0 (1-neck_head_ratio)*over_shoulder];

head_ind = find(strcmp(joint_labels,'jC1Head'));
% head_vec = rotate_vec_using_eul(head_vec_length,rotation_values(head_ind,:))
head_vec = rotate_vec_using_eul(head_vec,rotation_values(head_ind,:))

%% Upper Arms
arm_length = (arm_span-shoulder_width)/2; 

if npose
% uArmR_vec_length= arm_ratio*arm_length;
uArmR_vec = -[0 0 arm_ratio*arm_length];

else
% uArmR_vec_length= arm_ratio*arm_length;
uArmR_vec = -[0 arm_ratio*arm_length 0];
end

uArmR_ind = find(strcmp(joint_labels,'jRightShoulder'));
% uArmR_vec = rotate_vec_using_eul(uArmR_vec_length,rotation_values(uArmR_ind,:))
uArmR_vec = rotate_vec_using_eul(uArmR_vec,rotation_values(uArmR_ind,:))


if npose
% uArmL_vec_length= arm_ratio*arm_length;
uArmL_vec = -[0 0 arm_ratio*arm_length];

else
% uArmL_vec_length= arm_ratio*arm_length;
uArmL_vec = [0 arm_ratio*arm_length 0];

end
uArmL_ind = find(strcmp(joint_labels,'jLeftShoulder'));
% uArmL_vec = rotate_vec_using_eul(uArmL_vec_length,rotation_values(uArmL_ind,:))
uArmL_vec = rotate_vec_using_eul(uArmL_vec,rotation_values(uArmL_ind,:))


%% Lower Arms
if npose
% lArmR_vec_length= arm_ratio*arm_length;
lArmR_vec = -[0 0 arm_ratio*arm_length];

else
% lArmR_vec_length= arm_ratio*arm_length;
lArmR_vec = -[0 arm_ratio*arm_length 0];

end
lArmR_ind = find(strcmp(joint_labels,'jRightElbow'));
% lArmR_vec = rotate_vec_using_eul(lArmR_vec_length,rotation_values(lArmR_ind,:))
lArmR_vec = rotate_vec_using_eul(lArmR_vec,rotation_values(lArmR_ind,:))


if npose
% lArmL_vec_length= arm_ratio*arm_length;
lArmL_vec = -[0 0 arm_ratio*arm_length];

else
% lArmL_vec_length= arm_ratio*arm_length;
lArmL_vec = +[0 arm_ratio*arm_length 0];

end
lArmL_ind = find(strcmp(joint_labels,'jLeftElbow'));
% lArmL_vec = rotate_vec_using_eul(lArmL_vec_length,rotation_values(lArmL_ind,:))
lArmL_vec = rotate_vec_using_eul(lArmL_vec,rotation_values(lArmL_ind,:))

%% Hands

if npose
% handR_vec_length= (1-2*arm_ratio)*arm_length;
handR_vec = -[0 0 (1-2*arm_ratio)*arm_length];

else
% handR_vec_length= (1-2*arm_ratio)*arm_length;
handR_vec = -[0 (1-2*arm_ratio)*arm_length 0];
end
handR_ind = find(strcmp(joint_labels,'jRightWrist'));
% handR_vec = rotate_vec_using_eul(handR_vec_length,rotation_values(handR_ind,:))
handR_vec = rotate_vec_using_eul(handR_vec,rotation_values(handR_ind,:))


if npose
% handL_vec_length= (1-2*arm_ratio)*arm_length;
handL_vec = -[0 0 (1-2*arm_ratio)*arm_length];

else
% handL_vec_length= (1-2*arm_ratio)*arm_length;
handL_vec = [0 (1-2*arm_ratio)*arm_length 0];

end
handL_ind = find(strcmp(joint_labels,'jLeftWrist'));
% handL_vec = rotate_vec_using_eul(handL_vec_length,rotation_values(handL_ind,:))
handL_vec = rotate_vec_using_eul(handL_vec,rotation_values(handL_ind,:))

%% Bases info

root = character_root;

uLegR_base = root - pelv_vec; 
uLegL_base = root + pelv_vec;

lLegR_base = uLegR_base + uLegR_vec;
lLegL_base = uLegL_base + uLegL_vec;


FootR_base = lLegR_base + lLegR_vec;
FootL_base = lLegL_base + lLegL_vec;


BallR_base = FootR_base + FootR_vec;
BallL_base = FootL_base + FootL_vec;


L5_base = root;
L3_base = L5_base+L5_vec; 
T12_base = L3_base+L3_vec; 
T8_base = T12_base+T12_vec; 


ShoulderR_base = T8_base+T8_vec;
ShoulderL_base = T8_base+T8_vec;

neck_base = T8_base+T8_vec;
head_base = neck_base+neck_vec;

uArmR_base = ShoulderR_base+ShoulderR_vec;
uArmL_base = ShoulderL_base+ShoulderL_vec;

lArmR_base = uArmR_base+uArmR_vec;
lArmL_base = uArmL_base+uArmL_vec;

handR_base = lArmR_base+lArmR_vec;
handL_base = lArmL_base+lArmL_vec;



%% Build Vectors
figure(3)

subplot(1,2,2)
% rotate3d
ax = gca
if idx == 1
CameraLoc = camera_placement;
else
CameraLoc = ax.CameraPosition ;
end
pelvis = quiver3(root(1), root(2), root(3), pelv_vec(1), pelv_vec(2), pelv_vec(3),'LineWidth',pelvis_specs,'ShowArrowHead','off');
hold on
quiver3(root(1), root(2), root(3), -pelv_vec(1), -pelv_vec(2), -pelv_vec(3),'LineWidth',pelvis_specs,'ShowArrowHead','off','Color',pelvis.Color)
pelvis_marker = plot3(root(1), root(2), root(3),'oR')

quiver3(uLegR_base(1), uLegR_base(2), uLegR_base(3),uLegR_vec(1), uLegR_vec(2), uLegR_vec(3),'LineWidth',thigh_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(uLegR_base(1), uLegR_base(2), uLegR_base(3),'oR','MarkerSize',12,'MarkerFaceColor',pelvis_marker.Color)
quiver3(uLegL_base(1),uLegL_base(2),uLegL_base(3), uLegL_vec(1), uLegL_vec(2), uLegL_vec(3),'LineWidth',thigh_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(uLegL_base(1), uLegL_base(2), uLegL_base(3),'oR','MarkerSize',12,'MarkerFaceColor',pelvis_marker.Color)


quiver3(lLegR_base(1), lLegR_base(2), lLegR_base(3),lLegR_vec(1), lLegR_vec(2), lLegR_vec(3),'LineWidth',shin_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(lLegR_base(1), lLegR_base(2), lLegR_base(3),'oR','MarkerSize',12,'MarkerFaceColor',pelvis_marker.Color)
quiver3(lLegL_base(1), lLegL_base(2), lLegL_base(3),lLegL_vec(1), lLegL_vec(2), lLegL_vec(3),'LineWidth',shin_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(lLegL_base(1), lLegL_base(2), lLegL_base(3),'oR','MarkerSize',12,'MarkerFaceColor',pelvis_marker.Color)


quiver3(FootR_base(1), FootR_base(2), FootR_base(3),FootR_vec(1), FootR_vec(2), FootR_vec(3),'LineWidth',foot_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(FootR_base(1), FootR_base(2), FootR_base(3), 'oR','MarkerSize',12,'MarkerFaceColor',pelvis_marker.Color)
quiver3(FootL_base(1), FootL_base(2), FootL_base(3),FootL_vec(1), FootL_vec(2), FootL_vec(3),'LineWidth',foot_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(FootL_base(1), FootL_base(2), FootL_base(3), 'oR','MarkerSize',12,'MarkerFaceColor',pelvis_marker.Color)


quiver3(BallR_base(1), BallR_base(2), BallR_base(3),BallR_vec(1), BallR_vec(2), BallR_vec(3),'LineWidth',Ball_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(BallR_base(1), BallR_base(2), BallR_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)
quiver3(BallL_base(1), BallL_base(2), BallL_base(3),BallL_vec(1), BallL_vec(2), BallL_vec(3),'LineWidth',Ball_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(BallL_base(1), BallL_base(2), BallL_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)


quiver3(L5_base(1), L5_base(2), L5_base(3),L5_vec(1), L5_vec(2), L5_vec(3),'LineWidth',0.5*spine_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(L5_base(1), L5_base(2), L5_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)
quiver3(L3_base(1), L3_base(2), L3_base(3),L3_vec(1), L3_vec(2), L3_vec(3),'LineWidth',spine_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(L3_base(1), L3_base(2), L3_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)
quiver3(T12_base(1), T12_base(2), T12_base(3),T12_vec(1), T12_vec(2), T12_vec(3),'LineWidth',T12_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(T12_base(1), T12_base(2), T12_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)
quiver3(T8_base(1), T8_base(2), T8_base(3),T8_vec(1), T8_vec(2), T8_vec(3),'LineWidth',T8_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(T8_base(1), T8_base(2), T8_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)


quiver3(neck_base(1), neck_base(2), neck_base(3),neck_vec(1), neck_vec(2), neck_vec(3),'LineWidth',neck_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(neck_base(1), neck_base(2), neck_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)
quiver3(head_base(1), head_base(2), head_base(3), head_vec(1), head_vec(2), head_vec(3),'LineWidth',head_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(head_base(1), head_base(2), head_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)


quiver3(ShoulderR_base(1), ShoulderR_base(2), ShoulderR_base(3),ShoulderR_vec(1), ShoulderR_vec(2), ShoulderR_vec(3),'LineWidth',shoulder_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(ShoulderR_base(1), ShoulderR_base(2), ShoulderR_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)
quiver3(ShoulderL_base(1), ShoulderL_base(2), ShoulderL_base(3), ShoulderL_vec(1), ShoulderL_vec(2), ShoulderL_vec(3),'LineWidth',shoulder_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(ShoulderL_base(1), ShoulderL_base(2), ShoulderL_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)


quiver3(uArmR_base(1), uArmR_base(2), uArmR_base(3),uArmR_vec(1), uArmR_vec(2), uArmR_vec(3),'LineWidth',biceps_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(uArmR_base(1), uArmR_base(2), uArmR_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)
quiver3(uArmL_base(1), uArmL_base(2), uArmL_base(3), uArmL_vec(1), uArmL_vec(2), uArmL_vec(3),'LineWidth',biceps_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(uArmL_base(1), uArmL_base(2), uArmL_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)



quiver3(lArmR_base(1), lArmR_base(2), lArmR_base(3),lArmR_vec(1), lArmR_vec(2), lArmR_vec(3),'LineWidth',forearm_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(lArmR_base(1), lArmR_base(2), lArmR_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)
quiver3(lArmL_base(1), lArmL_base(2), lArmL_base(3), lArmL_vec(1), lArmL_vec(2), lArmL_vec(3),'LineWidth',forearm_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(lArmL_base(1), lArmL_base(2), lArmL_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)


quiver3(handR_base(1), handR_base(2), handR_base(3),handR_vec(1), handR_vec(2), handR_vec(3),'LineWidth',hand_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(handR_base(1), handR_base(2), handR_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)
quiver3(handL_base(1), handL_base(2), handL_base(3), handL_vec(1), handL_vec(2), handL_vec(3),'LineWidth',hand_specs,'ShowArrowHead','off','Color',pelvis.Color)
plot3(handL_base(1), handL_base(2), handL_base(3),'oR','MarkerFaceColor',pelvis_marker.Color)

figure_properties
% view(rotate_vec_using_eul(camera_placement,rotation_values(pelv_ind)))
view(CameraLoc)

hold off
drawnow