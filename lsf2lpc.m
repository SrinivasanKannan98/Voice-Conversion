function a = lsf2lpc(lsf)
%This function a 2D matrix of LSF parameters to Linear Prediction
%Coefficients
%'lsf' is expected to be a 2D matrix where each column of lsf has the lsf parameters corresponding to a frame
%'a' is a 2D matrix where each column has the LP coefficients for each
%frame

	a = zeros(size(lsf,1)+1,size(lsf,2));

	for i = 1 : size(lsf,2)

		temp = lsf(:,i);
		temp = sort(abs(temp),'ascend'); %Done to ensure that the LSF parameters are in an ascending order and positive
		replace = find(temp >= pi); %Checking if any of the LSF parameters are greater than pi

		if(~isempty(replace))      
            temp = temp ./ max(temp); %If any Line Spectral Frequency is above Pi all the Line Spectral Frequencies for that frame
                                      %are normalised by it and further
                                      %multiplied by Pi to ensure that the
                                      %LSFs lie between 0 and Pi
            temp = temp .* pi;
        end
        
		temp = sort(abs(temp),'ascend');
		a(:,i) = lsf2poly(temp); %In-Built MATLAB function used to convert LSF parameters to LP coefficients
		a(:,i) = polystab(a(:,i)); %In-Built MATLAB function used to ensure no pole lies outside the Unit Circle in the Z-Plane
 
	end

end