function[dw_target, dw_source, p, q] = dtws2(d_target,d_source)
%This function performs Dynamic Time Warping on the LP coefficients of the
%source and target utterances

    M = simmx(d_target,d_source);
    [p,q] = dp(1 - M);

    dt_mcep = d_target;
    ds_mcep = d_source;

    dt_mcep = dt_mcep';
    ds_mcep = ds_mcep';

    j = 2;
    pnew(1) = p(1);
    qnew(1) = q(1);

    for i = 2 : size(q,2)
        
      if(q(i) == q(i - 1)) 
        qnew(j) = q(i);
        pnew(j) = pnew(j - 1);
      elseif(p(i) == p(i - 1))
        qnew(j) = qnew(j - 1);
        pnew(j) = p(i);
      else
        qnew(j) = q(i);
        pnew(j) = p(i);
      end
      
      j = j + 1;
      
    end

    pnew = unique(pnew);
    qnew = unique(qnew);

    for i = 1 : length(qnew)
      dw_source(i,:) = ds_mcep(qnew(i),:);
    end

    for i = 1 : length(pnew)
      dw_target(i,:) = dt_mcep(pnew(i),:);
    end

    p = pnew;
    q = qnew;
    
    dw_source = dw_source';
    dw_target = dw_target';
end

function M = simmx(A,B)

    EA = sqrt(sum(A.^2));
    EB = sqrt(sum(B.^2));

    M = (A'*B)./(EA'*EB);

end

function [p,q,D] = dp(M)

    [r,c] = size(M);

    D = zeros(r+1, c+1);
    D(1,:) = NaN;
    D(:,1) = NaN;
    D(1,1) = 0;
    D(2:(r+1), 2:(c+1)) = M;

    phi = zeros(r,c);

    for i = 1 : r
      for j = 1 : c
        [dmax, tb] = min([D(i, j), D(i, j+1), D(i+1, j)]);
        D(i + 1,j + 1) = D(i + 1,j + 1) + dmax;
        phi(i,j) = tb;
      end
    end

    i = r; 
    j = c;
    p = i;
    q = j;
    while (i > 1 && j > 1)
        
      tb = phi(i,j);
      if (tb == 1)
        i = i - 1;
        j = j - 1;
      elseif (tb == 2)
        i = i - 1;
      elseif (tb == 3)
        j = j - 1;
      else    
        error;
      end
      
      p = [i,p];
      q = [j,q];
      
    end

    D = D(2:(r+1),2:(c+1));
    
end





