function synth = lpSynthesis(alpha,G,resi,fs,frameLen,hopLen,winName,preemphasise)
%This function performs LP synthesis using the parameters given as inputs
%to the function. Each column of 'alpha' contains the LP coefficients of
%the respective frame, 'G' holds the gain of each frame and 'resi' holds
%the excitation signal for each signal
%If 'preemphasis' = 1, Pre-Emphasis of the signal had been done during
%analysis, hence, de-emphasis is required

    if nargin < 5
        frameLen = floor(fs * 0.040);
    end
    if nargin < 6
        hopLen = floor(fs * 0.020);
    end
    
    if nargin < 7
        winName = "hann";
    end
    if nargin < 8
        preemphasise = 1;
    end
    
    syn_frame = []; %Holds the final synthesised speech
    
    for i = 1 : size(alpha,2)
        
        temp = filter(G(i),alpha(:,i),resi(:,i)); %Each frame is reconstructed using the IIR LP Synthesis filter
        syn_frame = [syn_frame temp(:)];
        
    end
    
    synth = OLA(syn_frame,frameLen,hopLen,winName); %The frames are combined using Overlap Add method to obtain the final speech signal
    
    %De-emphasis if necessary
    
    if(preemphasise == 1)
        synth = filter(1,[1 -0.9375],synth); %Should only be done if input was pre-emphasised and the filter coefficients should match
                                             %that of the Pre-emphasis filter
    end

end
        