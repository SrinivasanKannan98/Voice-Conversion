# Voice-Conversion

Two Methods of Voice Conversion have been proposed in this work

1) The First Method performs Pitch Modification on the Residual Signal of the given input signal during offline conversion
2) The Second Method performs Pitch Modification on the Input Signal itself during offline conversion

For testing Method 1, run 'Main_2.m' and for testing Method 2 run 'Main_3.m'
For selecting the Method of conversion at runtime, run 'Voice_Conversion.m'
For GUI app, run 'Voice_Conversion_GUI.m'

Voice Conversion can either be done for the given data, using trained models, or, done after training a new conversion model. For using the framework for a new pair of speakers, 
training data of atleast 60 seconds of parallel utterances from both speakers is required, which is broken down to audio files containing about 3 seconds of speech each. If 
the entire training utterance is used directly, Dynamic Time Warping used before training the Neural Network will take a significant amount of time.

Pre-Requisites:

MATLAB R2017 or above is required. Deep Learning Toolbox is to be installed. A CUDA enabled NVIDIA GPU is recommended. 
Praat.exe has to be installed in the project directory - https://www.fon.hum.uva.nl/praat/.

The database used for testing the framework is the CMU Arctic Database. Outputs of conversion between BDL(American Male) and SLT(American Female) and vice-versa, have been given.
