%%
function delay= getDelay(input_source,target_source,dt)

    cor=xcorr(input_source,target_source);
%     figure 
%     plot (cor)
    [~,index]=max(cor);
    delay=(size(input_source,1)-index)*dt;

end