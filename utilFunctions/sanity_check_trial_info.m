function bSameInfo = sanity_check_trial_info(past_trial_info,current_trial_info)
%check trials info to match 
rand_idx=randi(size(current_trial_info,2));
bSameInfo=false;

fn = fieldnames(past_trial_info{rand_idx})
for k=1:numel(fn)
    
   if past_trial_info{rand_idx}.(fn{k}) ~=   current_trial_info{rand_idx}.(fn{k})
       bSameInfo=false;
   else 
       bSameInfo=true;
   end
end

end

