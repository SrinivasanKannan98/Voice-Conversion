function [t, f0, avgF0, en] = F0Estimation(x, fs)

%This function uses the Autocorrelation method to return the Fundamental
%frequency of the input speech, based on the method proposed by Philipos C. Loizou

	ns = length(x);
	mu = mean(x);
	x = x - mu; 

	fRate = floor(120*fs/1000);  
	updRate = floor(110*fs/1000);
	nFrames = floor(ns/updRate)-1;

	f0 = zeros(1, nFrames);
	f01 = zeros(1, nFrames);

	k = 1;
	avgF0 = 0;
	m = 1;

	for i = 1 : nFrames

	  xseg = x(k : k + fRate - 1);
	  f01(i) = pcorr(fRate, fs, xseg); 
	  en = ((x(1 : nFrames)) .^ 2);
	  
	  if (i > 2 && nFrames > 3) 

	    z = f01(i - 2 : i); 
	    md = median(z);
	    f0(i - 2) = md;

	    if (md > 0)
	      avgF0 = avgF0 + md;
	      m = m + 1;
	    end

	  elseif (nFrames <= 3) 
	    f0(i) = a;
	    avgF0 = avgF0 + a;
	    m = m + 1;
	  end

	  k = k + updRate;

	end

	t = 1 : nFrames;
	t = 20 * t;

	if (m == 1)
	  avgF0 = 0;
	else
	  avgF0 = avgF0/(m - 1);
	end 

end

function [f0] = pcorr(len, fs, xseg)

	[bf0, af0] = butter(4, 900/(fs/2));
	xseg = filter(bf0, af0, xseg); 

	i13 = len/3;
	maxi1 = max(abs(xseg(1 : i13)));

	i23 = 2 * len/3;
	maxi2 = max(abs(xseg(i23 : len)));

	if (maxi1 > maxi2)
	  CL = 0.68 * maxi2;
	else 
	  CL = 0.68 * maxi1;
	end

	clip = zeros(len,1);
	ind1 = find(xseg >= CL);
	clip(ind1) = xseg(ind1) - CL;

	ind2 = find(xseg <= -CL);
	clip(ind2) = xseg(ind2)+CL;

	engy = norm(clip,2)^2;

	RR = xcorr(clip);
	m = len;

	LF = floor(fs/320); 
	HF = floor(fs/60);

	Rxx = abs(RR(m + LF : m + HF));
	[rmax, imax] = max(Rxx);

	imax = imax + LF;
	f0 = fs/imax;

	silence = 0.4*engy;

	if ((rmax > silence)  && (f0 > 60) && (f0 <= 320))
	 f0 = fs/imax;
	else 
	 f0 = 0;
	end

end