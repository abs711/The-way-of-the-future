function [] = generate_static_character(npose)

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


%% Pelvis
pelvy = [-hip_width/2:segment_increment:hip_width/2];
pelvx = zeros(size(pelvy));
pelvz = zeros(size(pelvy));


%% Thighs
Thigh_length = hip_height-knee_height; 
uLegRz = [-Thigh_length+pelvz(1):segment_increment:pelvz(1)];
uLegRx = pelvx(1)*ones(size(uLegRz));
uLegRy = pelvy(1)*ones(size(uLegRz));


uLegLz = [-Thigh_length+pelvz(end):segment_increment:pelvz(end)];
uLegLx = pelvx(end)*ones(size(uLegLz));
uLegLy = pelvy(end)*ones(size(uLegLz));


%% Shins
lLegRz = [-knee_height+uLegRz(1):segment_increment:uLegRz(1)];
lLegRx = uLegRx(1)*ones(size(lLegRz));
lLegRy = uLegRy(1)*ones(size(lLegRz));


lLegLz = [-knee_height+uLegLz(1):segment_increment:uLegLz(1)];
lLegLx = uLegLx(1)*ones(size(lLegRz));
lLegLy = uLegLy(1)*ones(size(lLegLz));


%% Feet, Balls
FootRx = [lLegRx(1):segment_increment:lLegRx(1)+Foot_length];
FootRy = lLegRy(1)*ones(size(FootRx));
FootRz = lLegRz(1)*ones(size(FootRx));

BallRx = [FootRx(end):segment_increment:FootRx(end)+Foot_length*ball_size];
BallRy = FootRy(end)*ones(size(BallRx));
BallRz = FootRz(end)*ones(size(BallRx));

FootLx = [lLegLx(1):segment_increment:lLegLx(1)+Foot_length];
FootLy = lLegLy(1)*ones(size(FootLx));
FootLz = lLegLz(1)*ones(size(FootLx));

BallLx = [FootLx(end):segment_increment:FootLx(end)+Foot_length*ball_size];
BallLy = FootLy(end)*ones(size(BallLx));
BallLz = FootLz(end)*ones(size(BallLx));

%% Spine
spine_length = shoulder_height-hip_height;

L5z = [pelvz(floor(length(pelvz)/2)):segment_increment:pelvz(floor(length(pelvz)/2))+0.25*spine_length];
L5x = pelvx(floor(length(pelvz)/2))*ones(size(L5z));
L5y = pelvy(floor(length(pelvz)/2))*ones(size(L5z));


L3z = [L5z(end):segment_increment:L5z(end)+0.25*spine_length];
L3x = L5x(end)*ones(size(L3z));
L3y = L5y(end)*ones(size(L3z));


T12z = [L3z(end):segment_increment:L3z(end)+0.25*spine_length];
T12x = L3x(end)*ones(size(T12z));
T12y = L3y(end)*ones(size(T12z));


T8z = [T12z(end):segment_increment:T12z(end)+0.25*spine_length];
T8x = T12x(end)*ones(size(T8z));
T8y = T12y(end)*ones(size(T8z));


%% Shoulder-Neck-Head
over_shoulder = height-shoulder_height;


ShoulderRy = [T8y(end):segment_increment:T8y(end)+0.5*shoulder_width];
ShoulderRx = T8x(end)*ones(size(ShoulderRy));
ShoulderRz = T8z(end)*ones(size(ShoulderRy));


ShoulderLy = [T8y(end)-0.5*shoulder_width:segment_increment:T8y(end)];
ShoulderLx = T8x(end)*ones(size(ShoulderLy));
ShoulderLz = T8z(end)*ones(size(ShoulderLy));


neckz = [T8z(end):segment_increment:T8z(end)+neck_head_ratio*over_shoulder];
neckx = T8x(end)*ones(size(neckz));
necky = T8y(end)*ones(size(neckz));


headz = [neckz(end):segment_increment:neckz(end)+(1-neck_head_ratio)*over_shoulder];
headx = neckx(end)*ones(size(headz));
heady = necky(end)*ones(size(headz));

%% Biceps-Triceps

arm_length = (arm_span-shoulder_width)/2; 

if npose
uArmRz = [ShoulderRz(end)-arm_ratio*arm_length:segment_increment:ShoulderRz(end)];
uArmRx = ShoulderRx(end)*ones(size(uArmRz));
uArmRy = ShoulderRy(end)*ones(size(uArmRz));

uArmLz = [ShoulderLz(1)-arm_ratio*arm_length:segment_increment:ShoulderLz(1)];
uArmLx = ShoulderLx(1)*ones(size(uArmLz));
uArmLy = ShoulderLy(1)*ones(size(uArmLz));
else
uArmRy = [ShoulderRy(end):segment_increment:ShoulderRy(end)+arm_ratio*arm_length];
uArmRx = ShoulderRx(end)*ones(size(uArmRy));
uArmRz = ShoulderRz(end)*ones(size(uArmRy));

uArmLy = [ShoulderLy(1)-arm_ratio*arm_length:segment_increment:ShoulderLy(1)];
uArmLx = ShoulderLx(1)*ones(size(uArmLy));
uArmLz = ShoulderLz(1)*ones(size(uArmLy));
end
%% Forearm
if npose
lArmRz = [uArmRz(1)-arm_ratio*arm_length:segment_increment:uArmRz(1)];
lArmRx = uArmRx(1)*ones(size(lArmRz));
lArmRy = uArmRy(1)*ones(size(lArmRz));

lArmLz = [uArmLz(1)-arm_ratio*arm_length:segment_increment:uArmLz(1)];
lArmLx = uArmLx(1)*ones(size(lArmLz));
lArmLy = uArmLy(1)*ones(size(lArmLz));
else
lArmRy = [uArmRy(end):segment_increment:uArmRy(end)+arm_ratio*arm_length];
lArmRx = uArmRx(end)*ones(size(lArmRy));
lArmRz = uArmRz(end)*ones(size(lArmRy));

lArmLy = [uArmLy(1)-arm_ratio*arm_length:segment_increment:uArmLy(1)];
lArmLx = uArmLx(1)*ones(size(lArmLy));
lArmLz = uArmLz(1)*ones(size(lArmLy));
end
%% Hands
if npose
handRz = [lArmRz(end)-(1-2*arm_ratio)*arm_length:segment_increment:lArmRz(end)];
handRx = lArmRx(end)*ones(size(handRz));
handRy = lArmRy(end)*ones(size(handRz));

handLz = [lArmLz(1)-(1-2*arm_ratio)*arm_length:segment_increment:lArmLz(1)];
handLx = lArmLx(1)*ones(size(handLz));
handLy = lArmLy(1)*ones(size(handLz));
else
handRy = [lArmRy(end):segment_increment:lArmRy(end)+(1-2*arm_ratio)*arm_length];
handRx = lArmRx(end)*ones(size(handRy));
handRz = lArmRz(end)*ones(size(handRy));

handLy = [lArmLy(1)-(1-2*arm_ratio)*arm_length:segment_increment:lArmLy(1)];
handLx = lArmLx(1)*ones(size(handLy));
handLz = lArmLz(1)*ones(size(handLy));
end

%% Build
figure(1)
line(pelvx,pelvy,pelvz,'LineWidth',pelvis_specs)
view(3)
hold on
plot3(0,0,0,'oR')
plot3(pelvx(end),pelvy(end),pelvz(end),'oR')
plot3(pelvx(1),pelvy(1),pelvz(1),'oR')




line(uLegRx,uLegRy,uLegRz,'LineWidth',thigh_specs)
plot3(uLegRx(1),uLegRy(1),uLegRz(1),'oR')
line(uLegLx,uLegLy,uLegLz,'LineWidth',thigh_specs)
plot3(uLegLx(1),uLegLy(1),uLegLz(1),'oR')


line(lLegRx,lLegRy,lLegRz,'LineWidth',shin_specs)
plot3(lLegRx(1),lLegRy(1),lLegRz(1),'oR')
line(lLegLx,lLegLy,lLegLz,'LineWidth',shin_specs)
plot3(lLegLx(1),lLegLy(1),lLegLz(1),'oR')


line(FootRx,FootRy,FootRz,'LineWidth',foot_specs)
plot3(FootRx(end),FootRy(end),FootRz(end),'oR')
line(FootLx,FootLy,FootLz,'LineWidth',foot_specs)
plot3(FootLx(end),FootLy(end),FootLz(end),'oR')

line(BallRx,BallRy,BallRz,'LineWidth',Ball_specs)
plot3(BallRx(end),BallRy(end),BallRz(end),'oR')
line(BallLx,BallLy,BallLz,'LineWidth',Ball_specs)
plot3(BallLx(end),BallLy(end),BallLz(end),'oR')



line(L5x,L5y,L5z,'LineWidth',0.5*spine_specs) 
plot3(L5x(end),L5y(end),L5z(end),'oR')
line(L3x,L3y,L3z,'LineWidth',spine_specs)
plot3(L3x(end),L3y(end),L3z(end),'oR')
line(T12x,T12y,T12z,'LineWidth',T12_specs)
plot3(T12x(end),T12y(end),T12z(end),'oR')
line(T8x,T8y,T8z,'LineWidth',T8_specs)
plot3(T8x(end),T8y(end),T8z(end),'oR')


line(neckx,necky,neckz,'LineWidth',neck_specs)
plot3(neckx(end),necky(end),neckz(end),'oR')


line(headx,heady,headz,'LineWidth',head_specs)
plot3(headx(end),heady(end),headz(end),'oR')


line(ShoulderRx,ShoulderRy,ShoulderRz,'LineWidth',shoulder_specs)
plot3(ShoulderRx(end),ShoulderRy(end),ShoulderRz(end),'oR')
line(ShoulderLx,ShoulderLy,ShoulderLz,'LineWidth',shoulder_specs)
plot3(ShoulderLx(1),ShoulderLy(1),ShoulderLz(1),'oR')


line(uArmRx,uArmRy,uArmRz,'LineWidth',biceps_specs)
plot3(uArmRx(end),uArmRy(end),uArmRz(end),'oR')
line(uArmLx,uArmLy,uArmLz,'LineWidth',biceps_specs)
plot3(uArmLx(1),uArmLy(1),uArmLz(1),'oR')


line(lArmRx,lArmRy,lArmRz,'LineWidth',forearm_specs)
plot3(lArmRx(end),lArmRy(end),lArmRz(end),'oR')
line(lArmLx,lArmLy,lArmLz,'LineWidth',forearm_specs)
plot3(lArmLx(1),lArmLy(1),lArmLz(1),'oR')


line(handRx,handRy,handRz,'LineWidth',hand_specs)
plot3(handRx(end),handRy(end),handRz(end),'oR')
line(handLx,handLy,handLz,'LineWidth',hand_specs)
plot3(handLx(1),handLy(1),handLz(1),'oR')

figure_properties

end