function [P, ela] = conj_RDA(Xs, classes, lowerdims, tensor_shape)
% =========================================================================
X_N = size(Xs, 2);
X = reshape(Xs, [tensor_shape X_N]);
Xs = mat_to_cell(X);
Xss = tensor(X);
lambda = 1e-1;


Xsample1 = Xs{1};
sizeX = size(Xsample1);
nmodes = length(sizeX);
nsamples = length(Xs);
nclasses = length(unique(classes));

Us = cell(1, nmodes);
for imode = 1:nmodes
    Us{imode} = orth(randn(sizeX(imode), lowerdims(imode)));
    %Us{imode} = randn(sizeX(imode), lowerdims(imode));
end

% calculate Xc - X for each class, where Xc is the class mean and X is the
% overall mean (stored in classmeandiffs) and Xcj - Xc where Xcj is the
% j'th observation from class c (stored in observationdiffs) and the number
% of observations from each class (stored in nis).
[~, ~, cmean_m_xmeans, xi_m_cmeans, nis] = classbased_differences(Xss, classes);
classmeandiffstensor = reshape(cmean_m_xmeans, [tensor_shape nclasses]);
observationdiffstensor = reshape(xi_m_cmeans, [tensor_shape nsamples]);
Rw = observationdiffstensor;
% ======================================== Rb =============================
nis = sqrt(nis);
tensor_nis = tenzeros([tensor_shape nclasses]);
cell_nis = cell(1,nclasses);
for n = 1:nclasses
    cell_nis{n} = nis(n)*ones(tensor_shape);
          switch length(tensor_shape)
            case 2
            tensor_nis(:,:,n) = cell_nis{n};
            case 3
            tensor_nis(:,:,:,n) = cell_nis{n};
            case 4
            tensor_nis(:,:,:,:,n) = cell_nis{n};
            case 5
            tensor_nis(:,:,:,:,:,n) = cell_nis{n};
            otherwise
            disp('tensor_shape is not between 2 and 5.');
         end
end
Rb = classmeandiffstensor.*tensor_nis;
% =========================================================================
Rb = Rb.data;
Rw = Rw.data;
P = kron(Us{1},Us{2});
Qb = reshape(Rb,prod(tensor_shape),nclasses);
Qw = reshape(Rw,prod(tensor_shape),nsamples);
%BB = Qw*Qw' - Qb*Qb' + lambda*eye(size(P, 1));
problem.M = stiefelfactory(size(P, 1), size(P, 2));
%problem.M = grassmannfactory(size(P, 1), size(P, 2));

ela = cputime;
function store = prepare(P, Qb, Qw, store)
        B = Qb*Qb';
        W = Qw*Qw';
        store.B = B;
        store.W = W;
        %store.BinvW = P'*W*P - P'*B*P;
        store.BinvW = P'*W*P - P'*B*P + lambda*sum(sum(abs(P)));
end
% Define the problem cost function and its Euclidean gradient.
        problem.cost  = @cost;
        function [f, store] = cost(P, store)
            store = prepare(P, Qb, Qw, store);
            f = trace(store.BinvW);
        end
     
        problem.grad = @grad;
        function [g, store] = grad(P, store)
            store = prepare(P, Qb, Qw, store);
            egrad = 2*store.W*P - 2*store.B*P;
            %store.egrad = egrad;
            store.egrad = egrad + lambda*sign(P);
            g = problem.M.egrad2rgrad(P, egrad);
        end
    
        
        
        % Solve.
        % options
        maxits = 1000;
        options.intialtau = -1;
        options.mxitr = maxits;
        options.record = 1;
        options.maxiter = maxits;
        % Minimize the cost function using Riemannian trust-regions
        [P,  ~] = conjugategradient(problem, P, options);
        
% =========================================================================
    ela=cputime-ela;
end