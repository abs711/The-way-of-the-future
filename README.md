# The-way-of-the-future



![The way of the future_compressed](https://user-images.githubusercontent.com/42185229/176984475-d5ac496e-7f14-48c0-b599-38e6ed130ac8.png)


A dataset of egocentric vision, eye-tracking and full body kinematics from about 24 hours of human locomotion in out-of-the-lab environments. The data can be accessed on figshare (Part 1: https://doi.org/10.6084/m9.figshare.c.6076607) and figshare+ (Part 2: https://doi.org/10.25452/figshare.plus.21761465).

Paper link: https://doi.org/10.1038/s41597-023-01932-7

## Dataset storage format
The data is stored in a hierarchical structure. Each leaf node (green, far right) is a file, stored in folders and
subfolders (blue boxes) as depicted here.

![data_format4](https://user-images.githubusercontent.com/42185229/176984711-26fe5781-f81b-446c-9b45-07764b4d80a6.png)



## Regarding the code
'csvSave_v5_kine_only.m' generates the processed .csv files with Kinematics data from the raw mvnx files.
'cvSave_v5_kinematics_and_vision.m' generates the same and also saves the id of synchronized vision frames.


Please refer to the visualizations below to verify the synchronization of Vision and Kinematics streams. These can be generated using the script 'custom_humanoid.m' in the Visualizations directory




https://user-images.githubusercontent.com/42185229/176592556-558e046a-59f0-4968-a83a-18c68e9d2e5f.mp4




https://user-images.githubusercontent.com/42185229/176586985-75a7d6a2-4445-4c51-b5f6-1d5c493f6bad.mp4




