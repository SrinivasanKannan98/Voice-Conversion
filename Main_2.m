clear; clc; close all;

%% Preparation Stage

fprintf("\nSelect the directory for the training source data\n"); %Commands for selecting the source folder for training data
dname_source = uigetdir;
temp = find(dname_source == '\');
nme = dname_source(temp(end) + 1 : end);
fprintf(nme + " " + "selected\n");
files_source = dir(dname_source);
fileIndex_source = find(~[files_source.isdir]);

fprintf("Select the directory for the training target data\n"); %Commands for selecting the target folder for training data
dname_target = uigetdir;
temp = find(dname_target == '\');
nme = dname_target(temp(end) + 1 : end);
fprintf(nme + " " + "selected\n");
files_target = dir(dname_target);
fileIndex_target = find(~[files_target.isdir]);

l_t = []; %Matrices to store training data for LSP mapping
l_s = [];
F0_s = []; %Arrays to hold the average fundamental frequency of each training utterance 
F0_t = [];

lpcOrder = input('Enter the order for LP Analysis {16/20/24/''Default''} : '); %Prediction Order
winName = "hann"; %Type of window used for Windowing to prevent Gibbs Oscillations
numTrainSamples = 50; % No. of Parallel Utterances used for training
preemphasise = 1; %If equal to 1 Pre-Emphasis is done on the signal during LP Analysis
method = 'Method 1';

switch lpcOrder %Decision of Number of Neurons in hidden layer of ANN based on the LPC Order
    case 16
        numHiddenNeurons = 27;
    case 20
        numHiddenNeurons = 34;
    case 24 
        numHiddenNeurons = 50;
    otherwise
        numHiddenNeurons = 50;
end

%% Training Stage

for k = 1 : numTrainSamples 
	
    [source,fs_source] = audioread(strcat(dname_source,'\',files_source(fileIndex_source(k)).name)); %Loading source training utterance from folder
    [target,fs_target] = audioread(strcat(dname_target,'\',files_target(fileIndex_target(k)).name)); %Loading target training utterance from folder

    frameLen = floor(fs_source * 0.030); %Frame Length of 30ms
    hopLen = floor(fs_source * 0.010);  %Hop Length of 10ms
    
    %% Feature Extraction Stage
    
    [a_source,g_source,r_source] = lpAnalysis(source,fs_source,lpcOrder,frameLen,hopLen,winName,preemphasise); %LP Analysis done for both training utterances 
                                                                                                               %to get a low dimensional parametric representation of speech
    [a_target,g_target,r_target] = lpAnalysis(target,fs_target,lpcOrder,frameLen,hopLen,winName,preemphasise);

    [lsf_target,lsf_source,~,~] = dtws(a_target,a_source); 
    %LP coefficients of each frame are converted to LSF parameters due to
    %better stability after mapping and time aligned using DTW
    
    resi_source = OLA(r_source,frameLen,hopLen,winName); %Excitation Signals of source and target utterance reconstructed using Overlap Add Method
    resi_target = OLA(r_target,frameLen,hopLen,winName);
    
    if(preemphasise == 1)
        resi_source = filter(1,[1 -0.9375],resi_source); %Compensate for the Pre-Emphasis done during LP Analysis
        resi_target = filter(1,[1 -0.9375],resi_target);
    end
    
    [~,~,f0s,~] = F0Estimation(resi_source,fs_source); %The fundamental frequency of each training utterance's residual signals
                                                       %is estimated and stored in an array
    [~,~,f0t,~] = F0Estimation(resi_target,fs_target); 
    
    l_s = [l_s lsf_source];
    l_t = [l_t lsf_target];
    
    F0_s = [F0_s f0s];
    F0_t = [F0_t f0t];
end

% Neural Network Training 

netLSF = newff(l_s,l_t,numHiddenNeurons); %A feedforward NN with 1 hidden layer of 50 neurons is used for the LSF mapping
netLSF.trainFcn = 'trainscg'; %The training algorithm is the scaled conjugate gradient method with error function as MSE
netLSF.trainParam.max_fail = 100000; %Max No. of validation failures
netLSF.trainParam.epochs = 100000; %Max No. of Epochs during Backpropagation
netLSF.trainParam.time = 420; %Training time is 7 mins
netLSF = train(netLSF,l_s,l_t,'UseGPU','yes'); %The network model is trained using GPU and stored in 'netLSF'

% Comment the training stage if trained models are available and uncomment this section

% fprintf("\nSelect the trained model\n");
% [file,path] = uigetfile('*.mat');
% load(fullfile(path,file));

%% Testing Stage

fprintf("Select a file for testing\n"); %Select the file for testing the conversion model
[file,path] = uigetfile('*.wav');
[test,fs_test] = audioread(fullfile(path,file));
temp = find(path == '\');
nme_test = path(temp(end - 1) + 1 : temp(end) - 1);

frameLen = floor(0.030 * fs_test);
hopLen = floor(0.010 * fs_test);

if(strcmp(lpcOrder,'Default'))
    if(strcmp(nme_test,'BDL'))
        lpcOrder = 16;
    elseif(strcmp(nme_test,'SLT'))
        lpcOrder = 20;
    else
        lpcOrder = 24;
    end
end
    
[a_test,g_test,r_test] = lpAnalysis(test,fs_test,lpcOrder,frameLen,hopLen,winName,preemphasise); %LP analysis done on the test utterance
lsf_test = lpc2lsf(a_test); %The LP coefficients are converted to LSF parameters for mapping using the Neural Net
lsf_morph = zeros(size(lsf_test));

for i = 1 : size(lsf_test,2)
    lsf_morph(:,i) = netLSF(lsf_test(:,i)); %Trained model used to convert the LSF parameters of the test signal to match that of the required signal
end

a_morph = lsf2lpc(lsf_morph); %Convert the modified LSF parameters back to LP Coefficients
%a_morph = coeff_interpolate(a_morph); %Done to create smooth transitions between frames by using Bilinear Interpolation
                                                                                                                 
%Pitch Modification

resi_test = OLA(r_test,frameLen,hopLen,winName); %Reconstructing the excitation signal of the test speech using Overlap-Add Method
if(preemphasise == 1)
    resi_test = filter(1,[1 -0.9375],resi_test); %Compensate the Pre-Emphasis done during LP Analysis
end
[~,~,f0s,~] = F0Estimation(resi_test,fs_test); %Estimating average fundamental frequency of the Test file
f0t = logLinearTransform(F0_s,F0_t,f0s); %Estimating the average fundamental frequency of the Expected Morphed speech
sf = f0t/f0s; %Scaling Factor for Pitch Scaling

resi_test = resi_test ./ (max(abs(resi_test) + 0.01));
[status,result] = system('delete_temp.bat'); %Calling a Batch Script to delete previously created temporary files
audiowrite('temp.wav',resi_test,fs_test); %Creating a temporary file to be used for pitch manipulation using Praat
[status,result] = system("Praat.exe --run PSOLA.praat" + " " + sf); %Using Praat to modify the Pitch Contour of the excitation signal using TD-PSOLA
[resi_morph,~] = audioread('temp_1.wav'); %Reading the Pitch Modified residual signal
if(preemphasise == 1)
    resi_morph = filter([1 -0.9375],1,resi_morph); %Pre-Emphasise the modified residual signal if necessary
end

%Speech Reconstruction

r_morph = segmnt(resi_morph,frameLen,hopLen); %Used to segment the Pitch Modified Excitation Signal into frames
window = windowChoice(winName,frameLen);
r_morph = r_morph .* repmat(window,1,size(r_morph,2)); %Breaking the Pitch Modified Excitation signal to frames and windowing them
morph = lpSynthesis(a_morph,g_test,r_morph,fs_test,frameLen,hopLen,winName,preemphasise); %Reconstruct the morphed utterance using LP Synthesis

%Post Processing

morph = filter([1 -1],[1 -0.99],morph); %Filter out low-freq components
morph = filter([1 1],[1 0.99],morph); %Filter out high-freq noise
soundsc(morph,fs_test);

%% Validation Stage

fprintf("Select a file for Validation\n");
[file,path] = uigetfile('*.wav');
[val,fs_val] = audioread(fullfile(path,file)); %Load the validation utterance

[a_val,g_val,r_val] = lpAnalysis(val,fs_val,lpcOrder,frameLen,hopLen,winName,preemphasise); %LP analysis done on the validation utterance
lsf_val = lpc2lsf(a_val); %The LP coefficients are converted to LSF parameters to test performance of the conversion 

%Mel Cepstrum Distortion
morph = morph ./ (max(abs(morph)) + 0.01);
m_morph = mfcc(val,fs_val,'NumCoeffs',25,'WindowLength',frameLen,'OverlapLength',(frameLen-hopLen))';
m_val = mfcc(morph,fs_test,'NumCoeffs',25,'WindowLength',frameLen,'OverlapLength',(frameLen-hopLen))';
[mm,mv,~,~] = dtws2(m_morph,m_val); %Accounting for timing errors
MCD = mean((10/log(10)) *sqrt(2 * sum(((mm - mv) .^ 2),1)));
fprintf("Mel Cepstrum Distortion = %f dB\n",MCD);

%LSF Performance Index

[lm,lv,~,~] = dtws2(lsf_morph,lsf_val); %Accounting for timing errors
dvm = mean(sqrt(mean(((lm - lv) .^ 2),1)));
[lv,lt,~,~] = dtws2(lsf_val,lsf_test);
dvt = mean(sqrt(mean(((lv - lt) .^ 2),1)));
PLSF = 1 - (dvm/dvt);
fprintf("LSF Performance Index = %f\n",PLSF);

%Waveforms and Spectrograms

%Waveforms
f = figure;
subplot(2,1,1);
plot([0 : 1/fs_test : (length(morph) - 1)/fs_test],morph);
xlabel('Time(s)');
ylabel('Amplitude');
title('Morphed Speech');
subplot(2,1,2);
plot([0 : 1/fs_val : (length(val) - 1)/fs_val],val);
xlabel('Time(s)');
ylabel('Amplitude');
title('Validation Speech');

%Spectrograms
f1 = figure;
subplot(2,1,1);
spectrogram(morph,128,[],[],fs_test,'yaxis'); colorbar
title('Spectrogram of Morphed Speech');
subplot(2,1,2);
spectrogram(val,128,[],[],fs_val,'yaxis'); colorbar
title('Spectrogram of Validation Speech');

%% Saving Data
saving = input('Do you wish to save the data ? {''Y'',''N''} : ');
if(strcmp(saving,'Y'))
    if(preemphasise == 1)
        pr = 'Pre-Emphasised';
    else
        pr = 'Not Pre-Emphasised';
    end

    if(strcmp(nme_test,'SLT'))
        conversion = 'F2M';
    else
        conversion = 'M2F';
    end

    audiowrite(strcat('Converted_',method,'_',pr,'_',conversion,'_LPC Order_',num2str(lpcOrder),'.wav'),...
        morph,fs_test);
    save(strcat('Trained Model_',method,'_',pr,'_',conversion,'_LPC Order_',num2str(lpcOrder)),'netLSF','F0_s','F0_t');
    saveas(f,strcat('Waveforms_',method,'_',pr,'_',conversion,'_LPC Order_',num2str(lpcOrder),'.jpg'));
    saveas(f1,strcat('Spectrograms_',method,'_',pr,'_',conversion,'_LPC Order_',num2str(lpcOrder),'.jpg'));

    if(strcmp(pr,'Pre-Emphasised'))
        pr = 'Yes';
    else
        pr = 'No';
    end
    fileID = fopen('Objective Evaluation Metrics.txt','a');
    fprintf(fileID,'\r\n%-6d  %-12s  %-4s  %-9d  %-9f  %-9f',str2num(method(end)),pr,conversion,lpcOrder,MCD,PLSF);
    fclose(fileID);
end

