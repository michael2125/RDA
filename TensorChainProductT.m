function X = TensorChainProductT(X,U,list)
for i = 1 : numel(list)
    X = ttm(X,U{list(i)}',list(i));
end