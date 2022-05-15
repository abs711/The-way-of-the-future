% This function functions like the build in MATLAB function csvwrite but
% allows a row of headers to be easily inserted
%
% known limitations
% 	The same limitation that apply to the data structure that exist with 
%   csvwrite apply in this function, notably:
%       m must not be a cell array
%
% Inputs
%   
%   filename    - Output filename
%   m           - array of data
%   headers     - a cell array of strings containing the column headers. 
%                 The length must be the same as the number of columns in m.
%   r           - row offset of the data (optional parameter)
%   c           - column offset of the data (optional parameter)
%
%
% Outputs
%   None
function csvwrite_with_headers_FRAMES(filename,m,headers)

data_with_headers = [headers; m];

%write Data to csv
writetable(cell2table(data_with_headers),filename,'writevariablenames',0)
