function win = windowChoice(inp,len)

%This function returns a window of type described in 'inp' of length
%'len' in the variable 'win'
    
    if(inp == "hann")
        win = hann(len);
    elseif(inp == "hamm")
        win = hamming(len);
    elseif(inp == "gauss")
        win = gausswin(len);
    elseif(inp == "rect")
        win = rectwin(len);
    end
    
end