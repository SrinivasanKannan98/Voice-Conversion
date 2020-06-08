function f0c = logLinearTransform(f0s,f0t,f0)

%This function uses a Log Linear Transform to estimate the fundamental
%frequency of the morphed speech given the average fundamental frequencies
%obtained during training and the average fundamental frequency of the test
%speech 

    g = find(f0s == 0);
    f0s(g) = [];
    f0t(g) = [];
    g = find(f0t == 0);
    f0s(g) = [];
    f0t(g) = [];
    
    f0s = log10(f0s); %Frequencies converted to log domain 
    f0t = log10(f0t);
    f0 = log10(f0);
    
    us = mean(f0s); %Estimation of statistical parameters
    ss = std(f0s);
    
    ut = mean(f0t);
    st = std(f0t);
    
    if(ss == 0)
        ss = 1;
        st = 1;
    end
    
    f0c = ut + ((st/ss) * (f0 - us)); %Log Linear Transformation
    
    f0c = 10 ^ f0c;
    
end
    
    
    