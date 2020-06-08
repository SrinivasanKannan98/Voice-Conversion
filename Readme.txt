Two methods of Voice Transformation have been proposed

The first method is implemented in 'Main_2.m' where pitch modification is done on the residual signal of the test utterance during the offline conversion

The second method is implemented in 'Main_3.m' where pitch modification is done on the actual test utterance during offline conversion

Both Methods have an option of doing the process with or without Pre-Emphasis. If the trained models are to be directly used, the respective portion on the code has to be uncommented suitably.

Dependencies: The project requires MATLAB R2016 or above with the Deep Learning Toolbox installed. A CUDA enabled NVIDIA GPU is required for quick training. It also requires Praat.exe to be installed for Windows in the folder of the project

The database used in this work is the CMU Arctic Database. The proposed algorithm has been demonstrated for conversion between BDL(American Male) and SLT(American Female) and vice-versa.