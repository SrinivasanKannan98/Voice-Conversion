function lsf = lpc2lsf(a)
%This function converts the LP coefficients of the matrix to Line Spectral Pairs 
%'a' is expected to be a 2D matrix where each column has the LP coefficients for each frame
%'lsf' is a 2D matrix where each column has the LSF parameters for each
%frame

	lsf = zeros(size(a,1) - 1,size(a,2));

	for i = 1 : size(a,2)
		lsf(:,i) = poly2lsf(a(:,i)); %In-built MATLAB function to convert the LP Coefficients to Line Spectral Pairs
	end

end
