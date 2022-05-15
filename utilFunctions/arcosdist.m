function arcosd = arcosdist(x1,x2)
for k = 1:size(x2,1)
arcosd(k) = real(2*acos(dot(x1,x2(k,:))));
end
end