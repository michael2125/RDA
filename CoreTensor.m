function [Core] = CoreTensor(X, U, tensor_shape)
% Notation
% Input       ...X (feature X nsamples)
%             ...U (1 X N cell project matrix)
%             ...tensor_shape is sample size
% Out         ...Core reduced core tensor
nsamples = size(X, 2);
N = length(tensor_shape);
X = reshape(X, [tensor_shape nsamples]);
X = tensor(X);
list = [1:N];
Core = TensorChainProductT(X,U,list);

end


