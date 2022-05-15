function rawRMSE = scaleErrors(normRMSE,minVal,maxVal)
rawRMSE=normRMSE.* abs(maxVal-minVal);

end