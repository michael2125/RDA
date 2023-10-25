function X = TensorChainProductXTT(X,U,n)
sizeU = numel(U);
for i = 1 : sizeU
        if i ~= n
            X = ttm(X,U{i}*U{i}',i);
        end
end