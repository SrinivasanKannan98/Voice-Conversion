function cof_smooth = coeff_interpolate(coeff)
%This function modifies the LP coefficients in a frame based on that of its
%immediate neighbours to create smooth transitions between succesive frames

	len = size(coeff,2);
	cof_smooth(:,1) = polystab(coeff(:,1));
	cof_smooth(:,len) = polystab(coeff(:,len));

	for i = 2 : len - 1

	  cf0 = coeff(:,i - 1);
	  cf1 = coeff(:,i);
	  cf2 = coeff(:,i + 1);
	  cof_smooth(:,i) = polystab(0.25*cf0 + 0.5*cf1 + 0.25*cf2);

	end

end


