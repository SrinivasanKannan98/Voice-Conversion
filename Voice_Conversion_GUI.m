classdef Voice_Conversion_GUI < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                     matlab.ui.Figure
        GridLayout                   matlab.ui.container.GridLayout
        LeftPanel                    matlab.ui.container.Panel
        VoiceConversionToolboxLabel  matlab.ui.control.Label
        TrainingDropDownLabel        matlab.ui.control.Label
        TrainingDropDown             matlab.ui.control.DropDown
        SelectTrainedModelButton     matlab.ui.control.Button
        SourceButton                 matlab.ui.control.Button
        TargetButton                 matlab.ui.control.Button
        StartTrainingButton          matlab.ui.control.Button
        LPCOrderDropDownLabel        matlab.ui.control.Label
        LPCOrderDropDown             matlab.ui.control.DropDown
        MethodDropDownLabel          matlab.ui.control.Label
        MethodDropDown               matlab.ui.control.DropDown
        TestSpeechButton             matlab.ui.control.Button
        PlayTestSpeechButton         matlab.ui.control.Button
        ValidationSpeechButton       matlab.ui.control.Button
        PlayValidationSpeechButton   matlab.ui.control.Button
        ConvertButton                matlab.ui.control.Button
        PlayConvertedSpeechButton    matlab.ui.control.Button
        ResetButton                  matlab.ui.control.Button
        RightPanel                   matlab.ui.container.Panel
        UIAxes                       matlab.ui.control.UIAxes
        UIAxes_2                     matlab.ui.control.UIAxes
        WaveformsLabel               matlab.ui.control.Label
        SpectrogramsLabel            matlab.ui.control.Label
    end

    % Properties that correspond to apps with auto-reflow
    properties (Access = private)
        onePanelWidth = 576;
    end

    
    properties (Access = public)
        training 
        trainedNet
        F0s
        F0t
        lpcOrder
        method
        dname_source
        nme_source
        files_source
        fileIndex_source
        dname_target
        nme_target
        files_target
        fileIndex_target
        winName
        preemphasise
        frameLen
        hopLen
        test
        fs_test
        nme_test
        val
        fs_val
        morph
        
    end
    

    % Callbacks that handle component events
    methods (Access = private)

        % Code that executes after component creation
        function startupFcn(app)
            %Default values for variables
            app.training = 1;
            app.lpcOrder = 16;
            app.method = 'Method 1';
            app.winName = "hann";
            app.preemphasise = 1;
            app.SelectTrainedModelButton.Visible = 'off';
            app.SourceButton.Visible = 'on';
            app.TargetButton.Visible = 'on';
            app.StartTrainingButton.Visible = 'on';
            app.UIAxes.Visible = 'off';
            app.UIAxes_2.Visible = 'off';
            app.WaveformsLabel.Visible = 'off';
            app.SpectrogramsLabel.Visible = 'off';

        end

        % Value changed function: TrainingDropDown
        function TrainingDropDownValueChanged(app, event)
            value = app.TrainingDropDown.Value;
            if(strcmp(value,'Train New Model'))
                app.training = 1;
                app.SelectTrainedModelButton.Visible = 'off';
                app.SourceButton.Visible = 'on';
                app.TargetButton.Visible = 'on';
                app.StartTrainingButton.Visible = 'on';
            else
                app.training = 0;
                app.SourceButton.Visible = 'off';
                app.TargetButton.Visible = 'off';
                app.StartTrainingButton.Visible = 'off';
                app.SelectTrainedModelButton.Visible = 'on';
            end
        end

        % Changes arrangement of the app based on UIFigure width
        function updateAppLayout(app, event)
            currentFigureWidth = app.UIFigure.Position(3);
            if(currentFigureWidth <= app.onePanelWidth)
                % Change to a 2x1 grid
                app.GridLayout.RowHeight = {480, 480};
                app.GridLayout.ColumnWidth = {'1x'};
                app.RightPanel.Layout.Row = 2;
                app.RightPanel.Layout.Column = 1;
            else
                % Change to a 1x2 grid
                app.GridLayout.RowHeight = {'1x'};
                app.GridLayout.ColumnWidth = {368, '1x'};
                app.RightPanel.Layout.Row = 1;
                app.RightPanel.Layout.Column = 2;
            end
        end

        % Button pushed function: SelectTrainedModelButton
        function SelectTrainedModelButtonPushed(app, event)
            [file,path] = uigetfile('*.mat');
            load(fullfile(path,file));
            app.trainedNet = netLSF;
            app.F0s = F0_s;
            app.F0t = F0_t;
        end

        % Value changed function: LPCOrderDropDown
        function LPCOrderDropDownValueChanged(app, event)
            value = app.LPCOrderDropDown.Value;
            if(strcmp(value,'16'))
                app.lpcOrder = 16;
            elseif(strcmp(value,'20'))
                app.lpcOrder = 20;
            elseif(strcmp(value,'24'))
                app.lpcOrder = 24;
            elseif(strcmp(value,'Default'))
                app.lpcOrder = 'Default';
            end
        end

        % Value changed function: MethodDropDown
        function MethodDropDownValueChanged(app, event)
            value = app.MethodDropDown.Value;
            if(strcmp(value,'Method 1'))
                app.method = 'Method 1';
            elseif(strcmp(value,'Method 2'))
                app.method = 'Method 2';
            elseif(strcmp(value,'Default'))
                app.method = 'Default';
            end
        end

        % Button pushed function: SourceButton
        function SourceButtonPushed(app, event)
            app.dname_source = uigetdir;
            temp = find(app.dname_source == '\');
            app.nme_source = app.dname_source(temp(end) + 1 : end);
            app.files_source = dir(app.dname_source);
            app.fileIndex_source = find(~[app.files_source.isdir]);
        end

        % Button pushed function: TargetButton
        function TargetButtonPushed(app, event)
            app.dname_target = uigetdir;
            temp = find(app.dname_target == '\');
            app.nme_target = app.dname_target(temp(end) + 1 : end);
            app.files_target = dir(app.dname_target);
            app.fileIndex_target = find(~[app.files_target.isdir]);
        end

        % Button pushed function: StartTrainingButton
        function StartTrainingButtonPushed(app, event)

            wb = waitbar(0, 'Feature Extraction in Progress ...','WindowStyle','modal');
            wbch = allchild(wb);
            jp = wbch(1).JavaPeer;
            jp.setIndeterminate(1);
            app.preemphasise = 1; %If equal to 1 Pre-Emphasis is done on the signal during LP Analysis
            l_t = []; %Matrices to store training data for LSP mapping
            l_s = [];
            app.F0s = []; %Arrays to hold the average fundamental frequency of each training utterance 
            app.F0t = [];
            numTrainSamples = 50;
            if(strcmp(app.lpcOrder,'Default'))
                if(strcmp(app.nme_source,'BDL'))
                    app.lpcOrder = 16;
                elseif(strcmp(app.nme_source,'SLT'))
                    app.lpcOrder = 20;
                else
                    app.lpcOrder = 24;
                end
            end
            
            if(strcmp(app.method,'Default'))
                if(strcmp(app.nme_source,'BDL'))
                    app.method = 'Method 1';
                else
                    app.method = 'Method 2';
                end
            end
        
            switch app.lpcOrder %Decision of Number of Neurons in hidden layer of ANN based on the LPC Order
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
        
                [source,fs_source] = audioread(strcat(app.dname_source,'\',app.files_source(app.fileIndex_source(k)).name)); %Loading source training utterance from folder
                [target,fs_target] = audioread(strcat(app.dname_target,'\',app.files_target(app.fileIndex_target(k)).name)); %Loading target training utterance from folder
        
                if(strcmp(app.method,'Method 2'))           
                    [~,~,f0s,~] = F0Estimation(source,fs_source); %The fundamental frequency of each training utterance pair is estimated and stored in an array
                    [~,~,f0t,~] = F0Estimation(target,fs_target); 
                    sf = f0t/f0s; %Scaling factor for pitch modification
                    [status,result] = system('delete_temp.bat'); %Use Praat to modify the Pitch of the source utterance to match that of the target utterance
                    audiowrite('temp.wav',source,fs_source);
                    [status,result] = system("Praat.exe --run PSOLA.praat" + " " + sf);
                    [source,~] = audioread('temp_1.wav');
                end
                
                app.frameLen = floor(fs_source * 0.030); %Frame Length of 30ms
                app.hopLen = floor(fs_source * 0.010);  %Hop Length of 10ms
            
                %% Feature Extraction Stage
        
                [a_source,g_source,r_source] = lpAnalysis(source,fs_source,app.lpcOrder,app.frameLen,app.hopLen,app.winName,app.preemphasise); %LP Analysis done for both training utterances 
                                                                                                                           %to get a low dimensional parametric representation of speech
                [a_target,g_target,r_target] = lpAnalysis(target,fs_target,app.lpcOrder,app.frameLen,app.hopLen,app.winName,app.preemphasise);
        
                [lsf_target,lsf_source,~,~] = dtws(a_target,a_source); 
                %LP coefficients of each frame are converted to LSF parameters due to
                %better stability after mapping and time aligned using DTW
        
                if(strcmp(app.method,'Method 1'))
                    resi_source = OLA(r_source,app.frameLen,app.hopLen,app.winName); %Excitation Signals of source and target utterance reconstructed using Overlap Add Method
                    resi_target = OLA(r_target,app.frameLen,app.hopLen,app.winName);
            
                    if(app.preemphasise == 1)
                        resi_source = filter(1,[1 -0.9375],resi_source); %Compensate for the Pre-Emphasis done during LP Analysis
                        resi_target = filter(1,[1 -0.9375],resi_target);
                    end
            
                    [~,~,f0s,~] = F0Estimation(resi_source,fs_source); %The fundamental frequency of each training utterance's residual signals
                                                                       %is estimated and stored in an array
                    [~,~,f0t,~] = F0Estimation(resi_target,fs_target); 
                end
                
                l_s = [l_s lsf_source];
                l_t = [l_t lsf_target];
        
                app.F0s = [app.F0s f0s];
                app.F0t = [app.F0t f0t];
                
            end            
            % Neural Network Training 
        
            netLSF = newff(l_s,l_t,numHiddenNeurons); %A feedforward NN with 1 hidden layer of 50 neurons is used for the LSF mapping
            netLSF.trainFcn = 'trainscg'; %The training algorithm is the scaled conjugate gradient method with error function as MSE
            netLSF.trainParam.max_fail = 100000; %Max No. of validation failures
            netLSF.trainParam.epochs = 100000; %Max No. of Epochs during Backpropagation
            netLSF.trainParam.time = 420; %Training time is 7 mins
            close(wb);
            netLSF = train(netLSF,l_s,l_t,'UseGPU','yes'); %The network model is trained using GPU and stored in 'netLSF'  
            app.trainedNet = netLSF;
            pause(2);

        end

        % Button pushed function: TestSpeechButton
        function TestSpeechButtonPushed(app, event)
            [file,path] = uigetfile('*.wav');
            [app.test,app.fs_test] = audioread(fullfile(path,file));
            temp = find(path == '\');
            app.nme_test = path(temp(end - 1) + 1 : temp(end) - 1);
            
            app.frameLen = floor(0.030 * app.fs_test);
            app.hopLen = floor(0.010 * app.fs_test);
            
            if(app.training ~= 1)
                if(strcmp(app.lpcOrder,'Default'))
                    if(strcmp(app.nme_test,'BDL'))
                        app.lpcOrder = 16;
                    elseif(strcmp(app.nme_test,'SLT'))
                        app.lpcOrder = 20;
                    else
                        app.lpcOrder = 24;
                    end
                end
                
                if(strcmp(app.method,'Default'))
                    if(strcmp(app.nme_test,'BDL'))
                        app.method = 'Method 1';
                    else
                        app.method = 'Method 2';
                    end
                end
            end
        end

        % Button pushed function: ValidationSpeechButton
        function ValidationSpeechButtonPushed(app, event)
            [file,path] = uigetfile('*.wav');
            [app.val,app.fs_val] = audioread(fullfile(path,file)); %Load the validation utterance            
        end

        % Button pushed function: PlayTestSpeechButton
        function PlayTestSpeechButtonPushed(app, event)
            soundsc(app.test,app.fs_test);
        end

        % Button pushed function: PlayValidationSpeechButton
        function PlayValidationSpeechButtonPushed(app, event)
            soundsc(app.val,app.fs_val);
        end

        % Button pushed function: ConvertButton
        function ConvertButtonPushed(app, event)
            
            wb = waitbar(0, 'Converting ...','WindowStyle','modal');
            wbch = allchild(wb);
            jp = wbch(1).JavaPeer;
            jp.setIndeterminate(1);
            
            if(strcmp(app.method,'Method 2'))
                %Pitch Modification
                [~,~,f0s,~] = F0Estimation(app.test,app.fs_test); %Estimate the average fundamental frequency of the test utterance
                f0t = logLinearTransform(app.F0s,app.F0t,f0s); %Estimate the average fundamental frequency of the expected morphed utterance
                sf = f0t/f0s; %Scaling Factor
            
                [status,result] = system('delete_temp.bat'); %Modify the average pitch frequency of the test utterance to match that of the expected morphed utterance
                audiowrite('temp.wav',app.test,app.fs_test);
                [status,result] = system("Praat.exe --run PSOLA.praat" + " " + sf);
                [app.test,~] = audioread('temp_1.wav');
            end
                
            [a_test,g_test,r_test] = lpAnalysis(app.test,app.fs_test,app.lpcOrder,app.frameLen,app.hopLen,app.winName,app.preemphasise); %LP analysis done on the test utterance
            lsf_test = lpc2lsf(a_test); %The LP coefficients are converted to LSF parameters for mapping using the Neural Net
            lsf_morph = zeros(size(lsf_test));
            
            for i = 1 : size(lsf_test,2)
                lsf_morph(:,i) = app.trainedNet(lsf_test(:,i)); %Trained model used to convert the LSF parameters of the test signal to match that of the required signal
            end
            
            a_morph = lsf2lpc(lsf_morph); %Convert the modified LSF parameters back to LP Coefficients
            %a_morph = coeff_interpolate(a_morph); %Done to create smooth transitions between frames by using Bilinear Interpolation
            
            if(strcmp(app.method,'Method 1'))
                %Pitch Modification
                resi_test = OLA(r_test,app.frameLen,app.hopLen,app.winName); %Reconstructing the excitation signal of the test speech using Overlap-Add Method
                if(app.preemphasise == 1)
                    resi_test = filter(1,[1 -0.9375],resi_test); %Compensate the Pre-Emphasis done during LP Analysis
                end
                [~,~,f0s,~] = F0Estimation(resi_test,app.fs_test); %Estimating average fundamental frequency of the Test file
                f0t = logLinearTransform(app.F0s,app.F0t,f0s); %Estimating the average fundamental frequency of the Expected Morphed speech
                sf = f0t/f0s; %Scaling Factor for Pitch Scaling
            
                resi_test = resi_test ./ (max(abs(resi_test) + 0.01));
                [status,result] = system('delete_temp.bat'); %Calling a Batch Script to delete previously created temporary files
                audiowrite('temp.wav',resi_test,app.fs_test); %Creating a temporary file to be used for pitch manipulation using Praat
                [status,result] = system("Praat.exe --run PSOLA.praat" + " " + sf); %Using Praat to modify the Pitch Contour of the excitation signal using TD-PSOLA
                [resi_morph,~] = audioread('temp_1.wav'); %Reading the Pitch Modified residual signal
                if(app.preemphasise == 1)
                    resi_morph = filter([1 -0.9375],1,resi_morph); %Pre-Emphasise the modified residual signal if necessary
                end
            
                %Speech Reconstruction
            
                r_morph = segmnt(resi_morph,app.frameLen,app.hopLen); %Used to segment the Pitch Modified Excitation Signal into frames
                window = windowChoice(app.winName,app.frameLen);
                r_morph = r_morph .* repmat(window,1,size(r_morph,2)); %Breaking the Pitch Modified Excitation signal to frames and windowing them
                app.morph = lpSynthesis(a_morph,g_test,r_morph,app.fs_test,app.frameLen,app.hopLen,app.winName,app.preemphasise); %Reconstruct the morphed utterance using LP Synthesis
            else
                app.morph = lpSynthesis(a_morph,g_test,r_test,app.fs_test,app.frameLen,app.hopLen,app.winName,app.preemphasise);
            end
            
            %Post Processing
            
            if(strcmp(app.method,'Method 2'))
                [~,~,f0s,~] = F0Estimation(app.morph,app.fs_test);
                sf = f0t/f0s; %Scaling Factor
                
                app.morph = app.morph ./ (max(abs(app.morph)) + 0.01);
                [status,result] = system('delete_temp.bat'); %Modify the average pitch frequency of the test utterance to match that of the expected morphed utterance
                audiowrite('temp.wav',app.morph,app.fs_test);
                [status,result] = system("Praat.exe --run PSOLA.praat" + " " + sf);
                [app.morph,~] = audioread('temp_1.wav');
            end
            
            app.morph = filter([1 -1],[1 -0.99],app.morph); %Filter out low-freq components
            app.morph = filter([1 1],[1 0.99],app.morph); %Filter out high-freq noise
            
            [a_val,g_val,r_val] = lpAnalysis(app.val,app.fs_val,app.lpcOrder,app.frameLen,app.hopLen,app.winName,app.preemphasise); %LP analysis done on the validation utterance
            lsf_val = lpc2lsf(a_val); %The LP coefficients are converted to LSF parameters to test performance of the conversion 
            
            %Mel Cepstrum Distortion
            app.morph = app.morph ./ (max(abs(app.morph)) + 0.01);
            m_morph = mfcc(app.val,app.fs_val,'NumCoeffs',25,'WindowLength',app.frameLen,'OverlapLength',(app.frameLen-app.hopLen))';
            m_val = mfcc(app.morph,app.fs_test,'NumCoeffs',25,'WindowLength',app.frameLen,'OverlapLength',(app.frameLen-app.hopLen))';
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
            close(wb);          
            PlayConvertedSpeechButtonPushed(app,1);
        end

        % Button pushed function: PlayConvertedSpeechButton
        function PlayConvertedSpeechButtonPushed(app, event)
           
            app.UIAxes.Visible = 'off';
            app.UIAxes_2.Visible = 'off';
            app.WaveformsLabel.Visible = 'off';
            app.SpectrogramsLabel.Visible = 'off';
       
            title(app.UIAxes,[]);
            xlabel(app.UIAxes,[]);
            ylabel(app.UIAxes,[]);
            app.UIAxes.XAxis.TickLabels = {};
            app.UIAxes.YAxis.TickLabels = {};
            
            f = figure('visible','off','Color','w');
            subplot(2,1,1);
            plot([0 : 1/app.fs_test : (length(app.morph) - 1)/app.fs_test],app.morph);
            xlabel('Time(s)');
            ylabel('Amplitude');
            title('Morphed Speech');
            subplot(2,1,2);
            plot([0 : 1/app.fs_val : (length(app.val) - 1)/app.fs_val],app.val);
            xlabel('Time(s)');
            ylabel('Amplitude');
            title('Validation Speech');
            set(gcf,'Position',app.UIAxes.Position);
            saveas(f,'W_temp.jpg');
            I = imshow('W_temp.jpg','Parent', app.UIAxes);
            app.UIAxes.XLim = [0 I.XData(2)];
            app.UIAxes.YLim = [0 I.YData(2)];
            
            title(app.UIAxes_2,[]);
            xlabel(app.UIAxes_2,[]);
            ylabel(app.UIAxes_2,[]);
            app.UIAxes_2.XAxis.TickLabels = {};
            app.UIAxes_2.YAxis.TickLabels = {};
            
            f1 = figure('visible','off','Color','w');
            subplot(2,1,1);
            spectrogram(app.morph,128,[],[],app.fs_test,'yaxis'); colorbar
            title('Spectrogram of Morphed Speech');
            subplot(2,1,2);
            spectrogram(app.val,128,[],[],app.fs_val,'yaxis'); colorbar
            title('Spectrogram of Validation Speech');
            set(gcf,'Position',app.UIAxes_2.Position);
            saveas(f1,'S_temp.jpg');
            I = imshow('S_temp.jpg','Parent', app.UIAxes_2);
            app.UIAxes_2.XLim = [0 I.XData(2)];
            app.UIAxes_2.YLim = [0 I.YData(2)];
            
            soundsc(app.morph,app.fs_test);
            pause(0.5);
            app.UIAxes.Visible = 'on';
            app.UIAxes_2.Visible = 'on';
            app.WaveformsLabel.Visible = 'on';
            app.SpectrogramsLabel.Visible = 'on';
            [~,~] = system('delete_temp_images.bat');
            
        end

        % Button pushed function: ResetButton
        function ResetButtonPushed(app, event)
            app.training = 1; 
            app.trainedNet = {};
            app.F0s = [];
            app.F0t = [];
            app.lpcOrder = 16;
            app.method = 'Method 1';
            app.dname_source = [];
            app.nme_source = [];
            app.files_source = [];
            app.fileIndex_source = [];
            app.dname_target = [];
            app.nme_target = [];
            app.files_target = [];
            app.fileIndex_target = [];
            app.winName = "hann";
            app.preemphasise = 1;
            app.frameLen = [];
            app.hopLen = [];
            app.test = [];
            app.fs_test = [];
            app.nme_test = [];
            app.val = [];
            app.fs_val = [];
            app.morph = [];           
            app.TrainingDropDown.Value = 'Train New Model';
            app.LPCOrderDropDown.Value = '16';
            app.MethodDropDown.Value = 'Method 1';
            close all;
            I = imshow([],'Parent', app.UIAxes);
            I = imshow([],'Parent', app.UIAxes_2);
            runStartupFcn(app, @startupFcn)
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.AutoResizeChildren = 'off';
            app.UIFigure.Position = [100 100 640 480];
            app.UIFigure.Name = 'UI Figure';
            app.UIFigure.SizeChangedFcn = createCallbackFcn(app, @updateAppLayout, true);

            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {368, '1x'};
            app.GridLayout.RowHeight = {'1x'};
            app.GridLayout.ColumnSpacing = 0;
            app.GridLayout.RowSpacing = 0;
            app.GridLayout.Padding = [0 0 0 0];
            app.GridLayout.Scrollable = 'on';

            % Create LeftPanel
            app.LeftPanel = uipanel(app.GridLayout);
            app.LeftPanel.Layout.Row = 1;
            app.LeftPanel.Layout.Column = 1;

            % Create VoiceConversionToolboxLabel
            app.VoiceConversionToolboxLabel = uilabel(app.LeftPanel);
            app.VoiceConversionToolboxLabel.FontWeight = 'bold';
            app.VoiceConversionToolboxLabel.Position = [106 450 155 22];
            app.VoiceConversionToolboxLabel.Text = 'Voice Conversion Toolbox';

            % Create TrainingDropDownLabel
            app.TrainingDropDownLabel = uilabel(app.LeftPanel);
            app.TrainingDropDownLabel.HorizontalAlignment = 'right';
            app.TrainingDropDownLabel.Position = [25 399 48 22];
            app.TrainingDropDownLabel.Text = 'Training';

            % Create TrainingDropDown
            app.TrainingDropDown = uidropdown(app.LeftPanel);
            app.TrainingDropDown.Items = {'Train New Model', 'Select Trained Model'};
            app.TrainingDropDown.ValueChangedFcn = createCallbackFcn(app, @TrainingDropDownValueChanged, true);
            app.TrainingDropDown.Position = [88 399 100 22];
            app.TrainingDropDown.Value = 'Train New Model';

            % Create SelectTrainedModelButton
            app.SelectTrainedModelButton = uibutton(app.LeftPanel, 'push');
            app.SelectTrainedModelButton.ButtonPushedFcn = createCallbackFcn(app, @SelectTrainedModelButtonPushed, true);
            app.SelectTrainedModelButton.Position = [219 399 129 22];
            app.SelectTrainedModelButton.Text = 'Select Trained Model';

            % Create SourceButton
            app.SourceButton = uibutton(app.LeftPanel, 'push');
            app.SourceButton.ButtonPushedFcn = createCallbackFcn(app, @SourceButtonPushed, true);
            app.SourceButton.Position = [16 338 136 22];
            app.SourceButton.Text = 'Source';

            % Create TargetButton
            app.TargetButton = uibutton(app.LeftPanel, 'push');
            app.TargetButton.ButtonPushedFcn = createCallbackFcn(app, @TargetButtonPushed, true);
            app.TargetButton.Position = [219 338 136 22];
            app.TargetButton.Text = 'Target';

            % Create StartTrainingButton
            app.StartTrainingButton = uibutton(app.LeftPanel, 'push');
            app.StartTrainingButton.ButtonPushedFcn = createCallbackFcn(app, @StartTrainingButtonPushed, true);
            app.StartTrainingButton.Position = [134 291 100 22];
            app.StartTrainingButton.Text = 'Start Training';

            % Create LPCOrderDropDownLabel
            app.LPCOrderDropDownLabel = uilabel(app.LeftPanel);
            app.LPCOrderDropDownLabel.HorizontalAlignment = 'right';
            app.LPCOrderDropDownLabel.Position = [16 243 63 22];
            app.LPCOrderDropDownLabel.Text = 'LPC Order';

            % Create LPCOrderDropDown
            app.LPCOrderDropDown = uidropdown(app.LeftPanel);
            app.LPCOrderDropDown.Items = {'16', '20', '24', 'Default'};
            app.LPCOrderDropDown.ValueChangedFcn = createCallbackFcn(app, @LPCOrderDropDownValueChanged, true);
            app.LPCOrderDropDown.Position = [94 243 58 22];
            app.LPCOrderDropDown.Value = '16';

            % Create MethodDropDownLabel
            app.MethodDropDownLabel = uilabel(app.LeftPanel);
            app.MethodDropDownLabel.HorizontalAlignment = 'right';
            app.MethodDropDownLabel.Position = [208 243 46 22];
            app.MethodDropDownLabel.Text = 'Method';

            % Create MethodDropDown
            app.MethodDropDown = uidropdown(app.LeftPanel);
            app.MethodDropDown.Items = {'Method 1', 'Method 2', 'Default'};
            app.MethodDropDown.ValueChangedFcn = createCallbackFcn(app, @MethodDropDownValueChanged, true);
            app.MethodDropDown.Position = [268 243 87 22];
            app.MethodDropDown.Value = 'Method 1';

            % Create TestSpeechButton
            app.TestSpeechButton = uibutton(app.LeftPanel, 'push');
            app.TestSpeechButton.ButtonPushedFcn = createCallbackFcn(app, @TestSpeechButtonPushed, true);
            app.TestSpeechButton.Position = [16 139 136 22];
            app.TestSpeechButton.Text = 'Test Speech';

            % Create PlayTestSpeechButton
            app.PlayTestSpeechButton = uibutton(app.LeftPanel, 'push');
            app.PlayTestSpeechButton.ButtonPushedFcn = createCallbackFcn(app, @PlayTestSpeechButtonPushed, true);
            app.PlayTestSpeechButton.Position = [219 139 136 22];
            app.PlayTestSpeechButton.Text = 'Play Test Speech';

            % Create ValidationSpeechButton
            app.ValidationSpeechButton = uibutton(app.LeftPanel, 'push');
            app.ValidationSpeechButton.ButtonPushedFcn = createCallbackFcn(app, @ValidationSpeechButtonPushed, true);
            app.ValidationSpeechButton.Position = [16 89 136 22];
            app.ValidationSpeechButton.Text = 'Validation Speech';

            % Create PlayValidationSpeechButton
            app.PlayValidationSpeechButton = uibutton(app.LeftPanel, 'push');
            app.PlayValidationSpeechButton.ButtonPushedFcn = createCallbackFcn(app, @PlayValidationSpeechButtonPushed, true);
            app.PlayValidationSpeechButton.Position = [217.5 89 137 22];
            app.PlayValidationSpeechButton.Text = 'Play Validation Speech';

            % Create ConvertButton
            app.ConvertButton = uibutton(app.LeftPanel, 'push');
            app.ConvertButton.ButtonPushedFcn = createCallbackFcn(app, @ConvertButtonPushed, true);
            app.ConvertButton.Position = [16 35 136 22];
            app.ConvertButton.Text = 'Convert';

            % Create PlayConvertedSpeechButton
            app.PlayConvertedSpeechButton = uibutton(app.LeftPanel, 'push');
            app.PlayConvertedSpeechButton.ButtonPushedFcn = createCallbackFcn(app, @PlayConvertedSpeechButtonPushed, true);
            app.PlayConvertedSpeechButton.Position = [216 35 142 22];
            app.PlayConvertedSpeechButton.Text = 'Play Converted Speech';

            % Create ResetButton
            app.ResetButton = uibutton(app.LeftPanel, 'push');
            app.ResetButton.ButtonPushedFcn = createCallbackFcn(app, @ResetButtonPushed, true);
            app.ResetButton.Position = [134 191 100 22];
            app.ResetButton.Text = 'Reset';

            % Create RightPanel
            app.RightPanel = uipanel(app.GridLayout);
            app.RightPanel.Layout.Row = 1;
            app.RightPanel.Layout.Column = 2;

            % Create UIAxes
            app.UIAxes = uiaxes(app.RightPanel);
            title(app.UIAxes, '')
            xlabel(app.UIAxes, '')
            ylabel(app.UIAxes, '')
            app.UIAxes.XLim = [0 1];
            app.UIAxes.YLim = [0 1];
            app.UIAxes.XColor = [0.902 0.902 0.902];
            app.UIAxes.XTick = [];
            app.UIAxes.YColor = [0.902 0.902 0.902];
            app.UIAxes.YTick = [];
            app.UIAxes.ZColor = [0.902 0.902 0.902];
            app.UIAxes.ZTick = [];
            app.UIAxes.Position = [7 264 257 185];

            % Create UIAxes_2
            app.UIAxes_2 = uiaxes(app.RightPanel);
            title(app.UIAxes_2, '')
            xlabel(app.UIAxes_2, '')
            ylabel(app.UIAxes_2, '')
            app.UIAxes_2.XLim = [0 1];
            app.UIAxes_2.YLim = [0 1];
            app.UIAxes_2.XColor = [0.902 0.902 0.902];
            app.UIAxes_2.XTick = [];
            app.UIAxes_2.YColor = [0.902 0.902 0.902];
            app.UIAxes_2.YTick = [];
            app.UIAxes_2.ZColor = [0.902 0.902 0.902];
            app.UIAxes_2.ZTick = [];
            app.UIAxes_2.Position = [7 28 257 185];

            % Create WaveformsLabel
            app.WaveformsLabel = uilabel(app.RightPanel);
            app.WaveformsLabel.FontWeight = 'bold';
            app.WaveformsLabel.Position = [24 450 70 22];
            app.WaveformsLabel.Text = 'Waveforms';

            % Create SpectrogramsLabel
            app.SpectrogramsLabel = uilabel(app.RightPanel);
            app.SpectrogramsLabel.FontWeight = 'bold';
            app.SpectrogramsLabel.Position = [24 229 86 22];
            app.SpectrogramsLabel.Text = 'Spectrograms';

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = Voice_Conversion_GUI

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            % Execute the startup function
            runStartupFcn(app, @startupFcn)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end