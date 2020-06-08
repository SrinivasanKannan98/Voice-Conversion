%This function can be used to reconstruct a signal using Overlap Add Synthesis

function recon = OLA(input,frameLen,hopLen,winName)

%input is expected to be a 2D matrix where each column of the matrix is an overlapped frameLen
%frameLen is the length of each overlapped frame as number of samples
%hopLen is amount of samples by which the window slides in each iteration

    L = size(input,2);
    recon_len = frameLen + (L - 1) * hopLen;
    recon = zeros(recon_len,1);
    window = windowChoice(winName,frameLen);
    
    for i = 1 : L
        recon(1 + (i - 1) * hopLen : frameLen + (i - 1) * hopLen) = recon(1 + (i - 1) * hopLen : ...
                        frameLen+(i - 1) * hopLen) + (input(:, i));
    end
    
    E = sum(window .* window);                  
    recon = recon .* hopLen/E;
                                      
end
   

    
            
    



