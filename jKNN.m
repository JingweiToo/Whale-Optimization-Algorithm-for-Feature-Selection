
function Acc = jKNN(feat,label,HO)
%---// Parameter setting for k-value of KNN //
k=5; 

xtrain = feat(HO.training==1,:);  ytrain = label(HO.training==1); 
xvalid = feat(HO.test==1,:);      yvalid = label(HO.test==1); 
Model  = fitcknn(xtrain,ytrain,'NumNeighbors',k); 
ypred  = predict(Model,xvalid);

num_valid = length(yvalid);
correct   = 0;
for i = 1:num_valid
  if isequal(yvalid(i),ypred(i))
    correct = correct + 1;
  end
end
Acc = 100 * (correct / num_valid); 
end

