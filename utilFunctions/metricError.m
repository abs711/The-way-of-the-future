function outputMetric = metricError(thisTest,thisPreds,metric_type)

num_trials =size(thisTest,1);
if num_trials==1
    if strcmp(metric_type,'rmse')
        outputMetric = sqrt(mean((thisTest-thisPreds).^2));
    else if strcmp(metric_type,'mse')
            outputMetric = mean((thisTest-thisPreds).^2);
        end
    end
    
else
    
    
    outputMetric=zeros(1,num_trials);
    for idx_trials=1:1:num_trials
        
        if strcmp(metric_type,'rmse')
            outputMetric(idx_trials) = sqrt(mean((thisTest(idx_trials,:)-thisPreds(idx_trials,:)).^2));
        else
            if strcmp(metric_type,'mse')
                
                outputMetric (idx_trials)= mean((thisTest(idx_trials,:)-thisPreds(idx_trials,:)).^2);
            end
            
        end
    end
    
end