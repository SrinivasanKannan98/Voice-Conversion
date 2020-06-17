clear ; clc;
lpcOrder = 24;
method = 'Method 2';
winName = "hann";
preemphasise = 1;

fprintf("\nSelect a Trained Model");
[file,path] = uigetfile('*.mat');
load(fullfile(path,file));

fprintf("\nSelect the directory for the source testing data\n"); %Commands for selecting the source folder for training data
dname_test = uigetdir;
temp = find(dname_test == '\');
nme_test = dname_test(temp(end) + 1 : end);
fprintf(nme_test + " " + "selected\n");
files_test = dir(dname_test);
fileIndex_test = find(~[files_test.isdir]);

fprintf("Select the directory for the target validation data\n"); %Commands for selecting the target folder for training data
dname_val = uigetdir;
temp = find(dname_val == '\');
nme_val = dname_val(temp(end) + 1 : end);
fprintf(nme_val + " " + "selected\n");
files_val = dir(dname_val);
fileIndex_val = find(~[files_val.isdir]);

if(strcmp(nme_test,'SLT'))
    conversion = 'F2M';
else
    conversion = 'M2F';
end

for k = 1 : 5
    
    [test,fs_test] = audioread(fullfile(dname_test,files_test(fileIndex_test(k)).name));
    [val,fs_val] = audioread(fullfile(dname_val,files_val(fileIndex_val(k)).name));
    
    frameLen = floor(0.030 * fs_test);
    hopLen = floor(0.010 * fs_test);
    
    if(strcmp(method,'Method 2'))
        %Pitch Modification
        [~,~,f0s,~] = F0Estimation(test,fs_test); %Estimate the average fundamental frequency of the test utterance
        f0t = logLinearTransform(F0_s,F0_t,f0s); %Estimate the average fundamental frequency of the expected morphed utterance
        sf = f0t/f0s; %Scaling Factor

        [status,result] = system('delete_temp.bat'); %Modify the average pitch frequency of the test utterance to match that of the expected morphed utterance
        audiowrite('temp.wav',test,fs_test);
        [status,result] = system("Praat.exe --run PSOLA.praat" + " " + sf);
        [test,~] = audioread('temp_1.wav');
    end

    [a_test,g_test,r_test] = lpAnalysis(test,fs_test,lpcOrder,frameLen,hopLen,winName,preemphasise); %LP analysis done on the test utterance
    lsf_test = lpc2lsf(a_test); %The LP coefficients are converted to LSF parameters for mapping using the Neural Net
    lsf_morph = zeros(size(lsf_test));

    for i = 1 : size(lsf_test,2)
        lsf_morph(:,i) = netLSF(lsf_test(:,i)); %Trained model used to convert the LSF parameters of the test signal to match that of the required signal
    end

    a_morph = lsf2lpc(lsf_morph); %Convert the modified LSF parameters back to LP Coefficients
    %a_morph = coeff_interpolate(a_morph); %Done to create smooth transitions between frames by using Bilinear Interpolation

    if(strcmp(method,'Method 1'))
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
    else
        morph = lpSynthesis(a_morph,g_test,r_test,fs_test,frameLen,hopLen,winName,preemphasise);
    end

    %Post Processing

    if(strcmp(method,'Method 2'))
        [~,~,f0s,~] = F0Estimation(morph,fs_test);
        sf = f0t/f0s; %Scaling Factor

        morph = morph ./ (max(abs(morph)) + 0.01);
        [status,result] = system('delete_temp.bat'); %Modify the average pitch frequency of the test utterance to match that of the expected morphed utterance
        audiowrite('temp.wav',morph,fs_test);
        [status,result] = system("Praat.exe --run PSOLA.praat" + " " + sf);
        [morph,~] = audioread('temp_1.wav');
    end

    morph = filter([1 -1],[1 -0.99],morph); %Filter out low-freq components
    morph = filter([1 1],[1 0.99],morph); %Filter out high-freq noise
    
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
    
    if(preemphasise == 1)
        pr = 'Pre-Emphasised';
    else
        pr = 'Not Pre-Emphasised';
    end
    
    audiowrite(strcat(num2str(k),'_Converted_',method,'_',pr,'_',conversion,'_LPC Order_',num2str(lpcOrder),'.wav'),...
    morph,fs_test);

    if(strcmp(pr,'Pre-Emphasised'))
        pr = 'Yes';
    else
        pr = 'No';
    end
    fileID = fopen('Subjective Evaluation Examples.txt','a');
    fprintf(fileID,'\r\n%-2d  %-6d  %-12s  %-4s  %-9d  %-9f  %-9f',k,str2num(method(end)),pr,conversion,lpcOrder,MCD,PLSF);
    fclose(fileID);
    
end

    
    