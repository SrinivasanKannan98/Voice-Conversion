#Praat script to modify the Pitch Contour of speech using TD-PSOLA

form scaleFactor
	real sf 1.6171198
endform

in = Read from file: "temp.wav"
selectObject: in
dur = Get total duration

manipulation = To Manipulation: 0.01, 75, 600
selectObject: manipulation

pitchTier = Extract pitch tier
selectObject: pitchTier
Multiply frequencies: 0, dur, sf

selectObject: manipulation
plusObject: pitchTier
Replace pitch tier

selectObject: manipulation
new_sound = Get resynthesis (overlap-add)

selectObject: new_sound
Save as WAV file: "temp_1.wav"