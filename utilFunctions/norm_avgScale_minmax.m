function [normed_data,min_val,max_val] = norm_avgScale_minmax(input_data,col_idx,range_min_max)
%normalizes input data in the range 0 and 1.
%
load 'avgScalers.mat' avgTrainMinMaxScaler headers_ang_vel_acc

min_val = avgTrainMinMaxScaler(1,col_idx)
max_val = avgTrainMinMaxScaler(2,col_idx)

numOfElements=numel(input_data);
numOfZeros = numel(find (input_data == 0));

ratioOfZeroElements=numOfZeros/numOfElements;
if (min_val == max_val) || ratioOfZeroElements > 0.95
    normed_data= zeros(size(input_data));
    
else
    normed_data=(input_data -min_val)./(max_val- min_val);
end


if range_min_max(1) ~=0
    range2 = range_min_max(2) - range_min_max(1);
    normed_data = (normed_data * range2) +  range_min_max(1);
end

end

