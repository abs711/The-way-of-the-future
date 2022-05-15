function [idx,C] = cluster_actions(data,num_clusters,algo,distance_function,varargin)
        
        if length(varargin)~=0
        custom_function = varargin{1};
        idxseed = find(strcmp(varargin,'seed')==1)+1;
        seed = varargin{idxseed} ;   
        rng(seed);
        end
        
if string(algo) == string('kmedoids') & string(distance_function) == string('custom_function')
    
    [idx,C] = kmedoids(data,num_clusters,'Distance',custom_function);

elseif string(algo) == string('kmedoids') & string(distance_function)~= string('custom_function')
    
    [idx,C] = kmedoids(data,num_clusters,'Distance',distance_function);    

elseif string(algo) == string('kmeans') & string(distance_function) == string('custom_function')

    [idx,C] = kmeans(data,num_clusters,'Distance',custom_function);

elseif string(algo) == string('kmeans') & string(distance_function) ~= string('custom_function')

    [idx,C] = kmeans(data,num_clusters,'Distance',distance_function);
    
end