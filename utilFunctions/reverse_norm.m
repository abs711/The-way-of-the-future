function [original_data] = reverse_norm(normed_data,min_val,max_val)
%normalizes input data in the range 0 and 1.
%


original_data =normed_data.*(max_val- min_val)+min_val;



end