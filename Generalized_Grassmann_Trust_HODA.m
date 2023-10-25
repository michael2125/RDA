function [P, ela] = Generalized_Grassmann_Trust_HODA(Xs, classes, lowerdims, tensor_shape)
% =========================================================================
X_N = size(Xs, 2);
XX = Xs;
X = reshape(Xs, [tensor_shape X_N]);
Xs = mat_to_cell(X);
Xss = tensor(X);


Xsample1 = Xs{1};
sizeX = size(Xsample1);
nmodes = length(sizeX);
nsamples = length(Xs);
nclasses = length(unique(classes));

Us = cell(1, nmodes);
for imode = 1:nmodes
    Us{imode} = randn(sizeX(imode), lowerdims(imode));
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
% P = khatrirao(Us{1}, Us{2});
Qb = reshape(Rb,prod(tensor_shape),nclasses);
Qw = reshape(Rw,prod(tensor_shape),nsamples);
RR = Qw*Qw' - Qb*Qb';
problem.M = grassmanngeneralizedfactory(size(P, 1), size(P, 2), RR);

ela = cputime;
function store = prepare(P, Qb, Qw, store)
        B = Qb*Qb';
        W = Qw*Qw';
        store.B = B;
        store.W = W;
        store.BinvW = P'*W*P - P'*B*P;
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
            store.egrad = egrad;
            g = problem.M.egrad2rgrad(P, egrad);
        end
    
        problem.hess = @hess;
        function [h, store] = hess(P, eta, store)
            store = prepare(P, Qb, Qw, store);
            hess = 2*store.W*eta - 2*store.B*eta;
            h = problem.M.ehess2rhess(P, store.egrad, hess, eta);
        end
        
        % Solve.
        % options
        options.maxiter = 200;
        options.maxinner = 30;
        options.maxtime = inf;
        options.tolgradnorm = 1e-5;
        options.Delta_bar = problem.M.typicaldist();
        % Minimize the cost function using Riemannian trust-regions
        P = trustregions(problem, [], options);     
% =========================================================================
    ela=cputime-ela;
end





