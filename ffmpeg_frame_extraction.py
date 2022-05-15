from PIL import Image
import os
import sys
import numpy as np
from glob import glob, iglob
pr_dir = 'E:/Unstructured_data/Unstructured_Data'
dir_list = os.listdir(pr_dir)
print('subjects = ', dir_list)
print('')
for folders in dir_list:
        print(folders)
        os.chdir(pr_dir+'/'+str(folders))
        sbdir_list = os.listdir('.')
        print(sbdir_list)
        try:
                os.chdir(pr_dir+'/'+str(folders)+'/'+'pupil')
        except FileNotFoundError:
                print('Warning: FileNotFound')
        sbsbdir_list = os.listdir('.')
        print(os.getcwd())
        print(sbsbdir_list)
        if 'junk' in sbsbdir_list:
                sbsbdir_list.remove('junk')
        for subfolders in sbsbdir_list:
                this_trial_path = pr_dir+'/'+str(folders)+'/'+'unstructured'+'/'+'00'+str(sbsbdir_list.index(subfolders)+1)
                this_frame_path = this_trial_path+'/'+'frames'
                try:
                        a = np.load(pr_dir+'/'+str(folders)+'/'+'pupil'+'/'+str(subfolders)+'/'+'exports'+'/'+'000'+'/world_timestamps.npy')
                except FileNotFoundError:
                        print('Junk Folder')
                
                try:
                        os.makedirs(this_frame_path)
                except FileExistsError:
                        print('Trial'+''+str(sbsbdir_list.index(subfolders)+1)+'/'+'frames'+''+'folder exists in unstructured:/')
                os.chdir(pr_dir+'/'+str(folders)+'/'+'pupil'+'/'+str(subfolders))#+'/'+'exports'+'/'+'000')################################
                print(os.getcwd())
                print(str(len(os.listdir(this_frame_path))),str(a.size))
                if len(os.listdir(this_frame_path))!= a.size:
                       os.system('ffmpeg -i world.mp4 -an -sn -vsync 0 -vf scale=320:240 '+this_frame_path+'\op-%08d.png') #JPGFRAMES IN RGB @ LOW RESOLUTION
                else:
                        print('Skipping trial','','00'+ str(sbsbdir_list.index(subfolders)+1))
                print('Number of frames extracted in trial','','00'+ str(sbsbdir_list.index(subfolders)+1),'=',str(len(os.listdir(this_frame_path))))
                print('Total frames in trial','','00'+ str(sbsbdir_list.index(subfolders)+1),'=',str(a.size))
                if len(os.listdir(this_frame_path)) != a.size:
                                  file = open(this_trial_path+'/'+'frames_dropped.txt', 'w')
                                  file.write(str(a.size-len(os.listdir(this_frame_path)))+' '+'frames dropped in this trial')
                                  file.close()
                else:
                        try:
                                os.remove(this_trial_path+'/'+'frames_dropped.txt')
                                print('ALL FRAMES PRESENT. REMOVING FRAMES DROPPED TXT FILE...')
                        except FileNotFoundError:
                                print('NO "FRAMES DROPPED" FLAG EXISTS')
                print('converting png to jpg')
                os.chdir(this_frame_path)
                png_list = glob('op-*.png')
                for pngs in png_list:
                        filename = pngs.split('.')[0]
                        filename = filename+ '.jpg'
                        im = Image.open(pngs)
                        rgb_im = im.convert('RGB')
                        rgb_im.save(filename)
                        os.remove(pngs)
