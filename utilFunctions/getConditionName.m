function [conditionName,trialNum] = getConditionName(this_filename,conditionList)
%gets activity of this file
%initialized to zero or false


firstC= min(strfind(this_filename,'C'));
conditionName=this_filename(firstC:firstC+2);

 if contains(conditionList,conditionName)
     disp("Error: Activity match not found")
 end
 
 firstT=min(strfind(this_filename,'T'));
 trialNum=this_filename(firstT+1:firstT+4);
 
end

