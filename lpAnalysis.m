function [alpha,G,resi] = lpAnalysis(x,fs,order,frameLen,hopLen,winName,preemphasise)

%This functions computes the LP coefficients for speech signal in 'x'
%sampled at 'fs'. The LPC coefficients are a short-time measure of
%the speech signal which describe the signal as the output of an all-pole
%filter. This all-pole filter provides a good description of the speech
%articulators
%
%'order' describes the predictor order, i.e., for each frame of the signal
%'order' + 1 coefficients are generated as a result of LP Analysis.
%'frameLen' describes the length of a frame in number of samples, 'hopLen' 
%decribes the hop length between succesive analysis frames in number of
%samples, 'winName' describes the type of window used for windowing
%Each column of 'alpha','G' and 'resi' are the respective parameters
%corresponding to each frame of speech to be analysed
%Each column of 'alpha' holds 'order' + 1 coefficients, each column of 'G'
%holds the gain of each frame and each column of 'resi' holds the
%residual/excitation signal of each frame
%'preemphasis' is used to decide if Pre-Emphasis of the signal is required,
%if = 1, Pre-Emphasis is applied

	if nargin < 3
		order = 20;
	end
	if nargin < 4
		frameLen = floor(fs * 0.040); %25ms frame length
    end   
	if nargin < 5
		hopLen = floor(fs * 0.020); %5ms hop
    end
    if nargin < 6
        winName = "hann";
    end
    if nargin < 7
        preemphasise = 1;
    end

    %Pre-emphasis if necessary, Pre-Emphasis is done on the signal to
    %compensate for 6dB/Octave rolloff that occurs due to radiation at the
    %mouth

    if(preemphasise == 1)        
        x = filter([1 -0.9375],1,x);
    end

	[x_buf,~] = buffer(x,frameLen,(frameLen - hopLen),'nodelay'); 
    %Frame Blocking of speech for short-time analysis, with parameters
    %given as input to this function, each column of 'x_buf' holds one
    %frame of the input
    
	window = windowChoice(winName,frameLen); %'window' holds a window of 
                                             %size 'frameLen'

	x_buf = x_buf .* repmat(window,1,size(x_buf,2)); %Windowing each frame
                                                     %to prevent Gibbs
                                                     %Oscillations

	alpha = []; 
	G = [];
	resi = [];

	for i = 1 : size(x_buf,2) %Iterate over all frames of the input

		[a,g] = lpc(x_buf(:,i),order); %In-Built MATLAB function to extract
                                       %LP Coefficients of a frame along
                                       %with Power of Error
	    clear isnan;
	    a(isnan(a)) = 0;
	    g(isnan(g)) = 1;
	    g(find(g == 0)) = 1; %#ok<FNDSB>

	   	g = sqrt(g);	
	   	r = filter(a,g,x_buf(:,i)); %Error/Residual Signal obtained by
                                    %filtering the windowed frame by the
                                    %FIR LP Analysis filter

	   	alpha(:,i) = a(:);
	   	G(i) = g;
	   	resi(:,i) = r(:);

	end

end
