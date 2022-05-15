function [activityName,idx_Activity,trial_num] = getActivityName(this_filename,activities_list)
%gets activity of this file
%initialized to zero or false

trial_num=this_filename(isstrprop(this_filename,'digit'));
filename_alphabets = this_filename(isstrprop(this_filename,'alpha'));

for idx_activity = 1:1:size(activities_list,2)
    
   if contains(filename_alphabets,activities_list{1,idx_activity},'IgnoreCase',true)
       activityName = activities_list{1,idx_activity};
       idx_Activity =  idx_activity;

       break; 
       
   else
       idx_Activity = 0;
       activityName='None';
   end
end


 if idx_Activity == 0
     disp("Error: Activity match not found")
 end
 
end

