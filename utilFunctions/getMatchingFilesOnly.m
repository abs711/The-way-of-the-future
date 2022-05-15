function listing = getMatchingFilesOnly(listing,matchingStr)
 % removes entries/file not matching string 
  num_files=size(listing,1);
    matched_idx=zeros(num_files,1);
    %keep only files with CHK
    matched_idx(cellfun(@(x) ~isempty(x),regexp({listing.name},matchingStr,'match')))=true;
    listing=listing(logical(matched_idx));
end

