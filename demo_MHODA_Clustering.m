
clear all
clc

addpath( genpath(pwd) );

Dataset_name = {'ssvepexo'};
%==========================================================================
for iDB = 1:length( Dataset_name )% DB is data base; and length(DB) is number of Data Base.
    
    % load data
    name = Dataset_name{ iDB };
    fprintf('Dataset Name:   %s\n', name);
    fprintf('Loading   ...');
    %load( name );% loading dataset
    load '/home/yin/disk2/MNE-ssvepexo-data/mat/subject12.mat';
    fea = fea*1e7;
    fprintf('Done\n');
    nclass = length(find(gnd==1));
    % =====================================================================
    type_method = 'MHODA'; % as type = 'NTD_LE' & options.alpha = 0, that is NTD algorithm
    tensor_shape    = [24 24];% the dimension for dataset dependent reshaped tensors
    lowerdims = [8 8];
    % =====================================================================
    runtimes = 2; % runing 10 times
    cross_num = 10;
    % =====================================================================
    for rn = 1 : runtimes
        indx = zeros(nclass*rn*2, 1);
        n = 2*rn +1;
        indx(:,1) = find(gnd<n);
        indx = indx(:);
        X = fea( indx, :);                         %===================%
        Y = gnd( indx, 1);                     %      ѡ��ѵ����    %
        t1 = clock;% count time
        for cros = 1 : cross_num
            train = X';                         %===================%
            train_label = Y;                     %      ѡ��ѵ����    %
            nsamples = size(X, 1);
            nClasses = length(unique(Y));
            % =============================================================
            [P, ela] = Trust_HODA(train, train_label, lowerdims, tensor_shape);
            %[P, ela] = Generalized_Stiefel_Trust_HODA(train, train_label, lowerdims, tensor_shape);
            %[P, ela] = Grassmann_Trust_HODA(train, train_label, lowerdims, tensor_shape);
            %[P, ela] = oblique_Trust_HODA(train, train_label, lowerdims, tensor_shape);
            %[P, ela] = Generalized_Grassmann_Trust_HODA(train, train_label, lowerdims, tensor_shape);
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
    save(['Results/', 'resC_', type_method,'', name ],  'results');
    clear results;
end

