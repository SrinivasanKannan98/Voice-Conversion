function segmented = segmnt(x,frameLen,hopLen)
%This function performs frame blocking and segmentation on the audio stored
%in 'x' and stores each frame as a column of 'segmented'

    x = x(:);    
    xlen = length(x);
    L = 1 + fix((xlen - frameLen)/hopLen);
    
    segmented = [];
    for l = 0 : L - 1      
        xw = x(1 + l * hopLen : frameLen + l * hopLen);
        segmented = [segmented xw(:)];       
    end
    
end
    