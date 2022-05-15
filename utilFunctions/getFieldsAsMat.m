function fieldsAsMat=getFieldsAsMat(dataStruct,inputFields,subField)



numTrials=numel(dataStruct.Index);
numFields=numel(inputFields);

fieldsAsMat = [];
for idxTrial=1:numTrials % cycle thru trials for rows
    thisTrialMat=[];
    for idxFields = 1 : numFields % cycle thru fields for cols
        
        thisTrialMat=[thisTrialMat,dataStruct.(inputFields{idxFields})(idxTrial).(subField)];
        
    end % fields/cols
    fieldsAsMat=[fieldsAsMat;thisTrialMat];
    
end % trials/rows

if sum(isnan(fieldsAsMat)) >0
    disp('Error Nan found')
end
end
