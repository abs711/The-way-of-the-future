function [] = generate_videostream(frame_list,ts,im_path)

figure(2)
subplot(1,2,1)
frame_num = string(abs(imag(frame_list(ts))));
missingzeros = repelem('0',(8-strlength(frame_num)));
frame_name = strcat('op-',missingzeros,string(abs(imag(frame_list(ts)))),'.jpg');
frame_loc = fullfile(im_path,frame_name);
imshow(imread(frame_loc))
end