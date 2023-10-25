
clear all
clc

addpath( genpath(pwd) );

Dataset_name = {'new_dense_ETH80'};

%=============================================
for iDB = 1:length( Dataset_name )% DB is data base; and length(DB) is number of Data Base.
    
    % load data
    name = Dataset_name{ iDB };
    fprintf('Dataset Name:   %s\n', name);
    fprintf('Loading   ...');
    load( name );% loading dataset
    fprintf('Done\n');
    nclass = length(find(gnd==1));
    % =====================================================================
    type_method = 'Conj_MHODA'; % as type = 'NTD_LE' & options.alpha = 0, that is NTD algorithm
    tensor_shape    = [32 32];% the dimension for dataset dependent reshaped tensors
    lowerdims = [8 8];
    % =====================================================================
    runtimes = 4; % runing 10 times
    cross_num = 10;
    % =====================================================================
    for rn = 1 : runtimes
        indx = zeros(nclass*rn*2, 1);
        n = 2*rn +1;
        indx(:,1) = find(gnd<n);
        indx = indx(:);
        X = fea( indx, :);                         %===================%
        Y = gnd( indx, 1);                     %      选择训练集    %
        t1 = clock;% count time
        for cros = 1 : cross_num
            train = X';                         %===================%
            train_label = Y;                     %      选择训练集    %
            nsamples = size(X, 1);
            nClasses = length(unique(Y));
            % =============================================================
            [P, ela] = Conj_HODA(train, train_label, lowerdims, tensor_shape);
            % ============================== train  =======================
            V = P'*train;
            label = litekmeans(V', nClasses,'Replicates', 20);
            label = bestMap(train_label, label);
            results.AC(rn, cros) = length(find(train_label == label))/length(label);
            results.NMI(rn, cros) = MutualInfo(train_label,label);
            iinf = cros + (rn - 1)*cross_num;
            results.rse{iinf} = ela;
            clear V label;
            
        end
          
    end
    results.method = type_method;
    save(['Results\', 'resC_', type_method,'', name ],  'results');
    clear results;
end

