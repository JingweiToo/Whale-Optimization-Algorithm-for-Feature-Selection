
function [sFeat,Sf,Nf,curve] = jWOA(feat,label,N,max_Iter,HO)
% Parameters
lb  = 0; 
ub  = 1;
b   = 1;

fun = @jFitnessFunction;
dim = size(feat,2); 
X   = zeros(N,dim); 
for i = 1:N
	for d = 1:dim
    X(i,d) = lb + (ub - lb) * rand();
	end
end
fit  = zeros(1,N);
fitG = inf;
for i = 1:N
  fit(i) = fun(feat,label,(X(i,:) > 0.5),HO);
  if fit(i) < fitG
    fitG = fit(i); 
    Xgb  = X(i,:);
  end
end

curve = inf;
t = 1; 
%---Iteration start---------------------------------------------------
while t <= max_Iter
  a = 2 - t * (2 / max_Iter);
  for i = 1:N
    A = 2 * a * rand() - a;
    C = 2 * rand();
    p = rand(); 
    l = -1 + 2 * rand();  
    if p  < 0.5
      if abs(A) < 1
        for d = 1:dim
          Dx     = abs(C * Xgb(d) - X(i,d));
          X(i,d) = Xgb(d) - A * Dx;
        end
      elseif abs(A) >= 1
        for d = 1:dim
          k      = randi([1,N]);
          Dx     = abs(C * X(k,d) - X(i,d));
          X(i,d) = X(k,d) - A * Dx;
        end
      end
    elseif p >= 0.5
      for d = 1:dim
        dist   = abs(Xgb(d) - X(i,d));
        X(i,d) = dist * exp(b * l) * cos(2 * pi * l) + Xgb(d);
      end
    end
    XB = X(i,:);  XB(XB > ub) = ub;  XB(XB < lb) = lb; 
    X(i,:) = XB;
  end
  for i = 1:N
    fit(i) = fun(feat,label,(X(i,:) > 0.5),HO);
    if fit(i) < fitG
      fitG = fit(i);
      Xgb  = X(i,:);
    end
  end
  curve(t) = fitG;
  fprintf('\nIteration %d Best (WOA)= %f',t,curve(t))
  t = t + 1;
end
Pos   = 1:dim;
Sf    = Pos((Xgb > 0.5) == 1);
Nf    = length(Sf);
sFeat = feat(:,Sf); 
end




