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

lpcOrder = 24; %Prediction Order
winName = "hann"; %Type of window used for Windowing to prevent Gibbs Oscillations
numTrainSamples = 50; % No. of Parallel Utterances used for training
preemphasise = 0; %If equal to 1 Pre-Emphasis is done on the signal during LP Analysis

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
    
    [~,~,f0s,~] = F0Estimation(source,fs_source); %The fundamental frequency of each training utterance pair is estimated and stored in an array
    [~,~,f0t,~] = F0Estimation(target,fs_target); 
    multiply = f0t/f0s; %Scaling factor for pitch modification
    
    [status,result] = system('delete_temp.bat'); %Use Praat to modify the Pitch of the source utterance to match that of the target utterance
    audiowrite('temp.wav',source,fs_source);
    [status,result] = system("Praat.exe --run PSOLA.praat" + " " + multiply);
    [source,~] = audioread('temp_1.wav');
    
    %% Feature Extraction Stage
    
    [a_source,g_source,r_source] = lpAnalysis(source,fs_source,lpcOrder,frameLen,hopLen,winName,preemphasise); %LP Analysis done for both training data 
                                                                                                               %to get parametric representation of speech
    [a_target,g_target,r_target] = lpAnalysis(target,fs_target,lpcOrder,frameLen,hopLen,winName,preemphasise);
    
    [lsf_target,lsf_source,~,~] = dtws(a_target,a_source); 
    %LP coefficients of each frame are converted to LSF parameters due to
    %better stability after mapping and time aligned using DTW
    
    l_s = [l_s lsf_source];
    l_t = [l_t lsf_target];
    
    F0_s = [F0_s f0s];
    F0_t = [F0_t f0t];
end

%% Neural Network Training 

netLSF = newff(l_s,l_t,numHiddenNeurons); %A feedforward NN with 1 hidden layer of 50 neurons is used for the LSF mapping
netLSF.trainFcn = 'trainscg'; %The training algorithm is the scaled conjugate gradient method with error function as MSE
netLSF.trainParam.max_fail = 100000; %Max No. of validation failures
netLSF.trainParam.epochs = 100000; %Max No. of Epochs during Backpropagation
netLSF.trainParam.time = 420; %Training time is 7 mins
netLSF = train(netLSF,l_s,l_t,'UseGPU','yes'); %The network model is trained using GPU and stored in 'netLSF'

%% Comment the training stage if trained models are available and uncomment this section

% fprintf("\nSelect the trained model\n");
% [file,path] = uigetfile('*.mat');
% load(fullfile(path,file));
% 
% fprintf("Select the fundamental frequency array\n");
% [file,path] = uigetfile('*.mat');
% load(fullfile(path,file));

%% Testing Stage

fprintf("Select a file for testing\n"); %Select the file for testing the conversion model
[file,path] = uigetfile('*.wav');
[test,fs_test] = audioread(fullfile(path,file));

frameLen = floor(0.030 * fs_test);
hopLen = floor(0.010 * fs_test);
    
%Pitch Modification

[~,~,f0s,~] = F0Estimation(test,fs_test); %Estimate the average fundamental frequency of the test utterance
f0t = logLinearTransform(F0_s,F0_t,f0s); %Estimate the average fundamental frequency of the expected morphed utterance
sf = f0t/f0s; %Scaling Factor

[status,result] = system('delete_temp.bat'); %Modify the average pitch frequency of the test utterance to match that of the expected morphed utterance
audiowrite('temp.wav',test,fs_test);
[status,result] = system("Praat.exe --run PSOLA.praat" + " " + sf);
[test,~] = audioread('temp_1.wav');

[a_test,g_test,r_test] = lpAnalysis(test,fs_test,lpcOrder,frameLen,hopLen,winName,preemphasise); %LP analysis done on the test utterance
lsf_test = lpc2lsf(a_test); %The LP coefficients are converted to LSF parameters for mapping using the Neural Net
lsf_morph = zeros(size(lsf_test));

for i = 1 : size(lsf_test,2)
    lsf_morph(:,i) = netLSF(lsf_test(:,i)); %Trained model used to convert the LSF parameters of the test signal to match that of the required signal
end

a_morph = lsf2lpc(lsf_morph); %Convert the modified LSF parameters back to LP Coefficients
%a_morph = coeff_interpolate(a_morph); %Done to create smooth transitions between frames by using Bilinear Interpolation
                                                                               
%Speech Reconstruction

morph = lpSynthesis(a_morph,g_test,r_test,fs_test,frameLen,hopLen,winName,preemphasise); %Reconstruct the morphed utterance using LP Synthesis
morph = morph ./ (max(abs(morph)) + 0.01);

%Post Processing

[~,~,f0s,~] = F0Estimation(morph,fs_test);
sf = f0t/f0s; %Scaling Factor

[status,result] = system('delete_temp.bat'); %Modify the average pitch frequency of the test utterance to match that of the expected morphed utterance
audiowrite('temp.wav',morph,fs_test);
[status,result] = system("Praat.exe --run PSOLA.praat" + " " + sf);
[morph,~] = audioread('temp_1.wav');
morph = filter([1 -1],[1 -0.99],morph); %Filter out low-freq components
morph = filter([1 1],[1 0.99],morph); %Filter out high-freq noise
morph = morph ./ (max(abs(morph)) + 0.01);

if(preemphasise == 1)
    pr = "Pre-Emphasised";
else
    pr = "Not Pre-Emphasised";
end
temp = find(path == '\');
nme = path(temp(end - 1) + 1 : temp(end) - 1);
if(strcmp(nme,'SLT'))
    conv = "F2M";
else
    conv = "M2F";
end

audiowrite(strcat('Converted_Method 2_',pr,'_',conv,'_',num2str(lpcOrder),'.wav'),...
    morph,fs_test);
soundsc(morph,fs_test);

%% Validation Stage

[file,path] = uigetfile('*.wav');
[val,fs_val] = audioread(fullfile(path,file)); %Load the validation utterance

% N_min = min(length(morph),length(val)); 
% morph = morph';
% val = val';
% morph = morph(1 : N_min); 
% val = val(1 : N_min);
% morph = morph';
% val = val'; %Aligning the number of samples of both the morphed and validation utterances

[a_val,g_val,r_val] = lpAnalysis(val,fs_val,lpcOrder,frameLen,hopLen,winName,preemphasise); %LP analysis done on the validation utterance
lsf_val = lpc2lsf(a_val); %The LP coefficients are converted to LSF parameters to test performance of the conversion 

%Mel Cepstrum Distortion

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

save('Trained Model_Method 2_' + pr + '_' + conv + '_LPC Order' + ' ' + num2str(lpcOrder),'netLSF');
save('F0 Array_Method 2_' + pr + '_' + conv + '_LPC Order' + ' ' + num2str(lpcOrder),'F0_s','F0_t');
save('Results_Method 2_' + pr + '_' + conv + '_LPC Order' + ' ' + num2str(lpcOrder) + '.txt','MCD','PLSF','-ascii');
saveas(f,'Waveforms_Method 2_' + conv + '_' + pr + '_LPC Order' + ' ' + num2str(lpcOrder) + '.jpg');
saveas(f1,'Spectrograms_Method 2_' + conv + '_' + pr + '_LPC Order' + ' ' + num2str(lpcOrder) + '.jpg');