function X = TensorChainProductNT(X,U,n)
sizeU = numel(U);
for i = 1 : sizeU
        if i ~= n
            X = ttm(X,U{i}',i);
        end
end