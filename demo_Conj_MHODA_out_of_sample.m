
clear all
clc

addpath( genpath(pwd) );

Dataset_name = {'new_dense_ETH80'};

%=============================================
cross_num = 3; % k-CV
for iDB = 1:length( Dataset_name )% DB is data base; and length(DB) is number of Data Base.
    
    % load data
    name = Dataset_name{ iDB };
    fprintf('Dataset Name:   %s\n', name);
    fprintf('Loading   ...');
    load( name );% loading dataset
    fprintf('Done\n');
    nClasses = length(unique(gnd));
    X = fea;
    Y = gnd;
    % =====================================================================
    type_method = 'Conj_MHODA'; % as type = 'NTD_LE' & options.alpha = 0, that is NTD algorithm
    tensor_shape    = [32 32];% the dimension for dataset dependent reshaped tensors
    lowerdims = [8 8];
    % =====================================================================
    runtimes = 10; % runing 10 times
    % =====================================================================
    for rn = 1 : runtimes
        indices = kFoldCV( Y, cross_num,  nClasses);
        % generates cross-validation indices
        t1 = clock;% count time
        for cros = 1 : cross_num
            fprintf('Starting cross validation: %dth\n', cros );
            %=============== test dataset and train dataset =====================%
            test_indx = (indices == cros);
            train_indx = ~test_indx;
            
            train = X( train_indx, :);                         %===================%
            train_label = Y( train_indx );                     %      选择训练集    %
            %                                                  %===================%
            
            test = X( test_indx, :);                       %===================%
            test_label = Y(test_indx);                         %     选择测试集     %
            test_nsamples = size(test, 1);
            train_nsamples = size(train, 1);
            % =============================================================
            train = train';
            test = test';
            T_nClasses = length(unique(test_label));
            R_nClasses = length(unique(train_label));
            % =================================================================
            [P, outputs, ela] = Conj_HODA(train, train_label, lowerdims, tensor_shape);
            % ============================== train  =======================
            V = P'*train;
            R_label = litekmeans(V', R_nClasses,'Replicates', 20);
            R_label = bestMap(train_label, R_label);
            results.train_AC(cros, rn) = length(find(train_label == R_label))/length(train_label);
            results.train_NMI(cros, rn) = MutualInfo(train_label,R_label);
            iinf = cros + (rn - 1)*cross_num;
            results.rse{iinf} = ela;
            clear V R_label;
            % ============================== test  ========================
            V = P'*test;
            % =============================================================
            cros_num = 5;
            indicess = kFoldCV( test_label, cros_num, T_nClasses );
            % generates cross-validation indices
            XX = V';
            for cro = 1 : cros_num
                fprintf('Starting cross validation: %dth\n', cro );
                %=============== test dataset and train dataset =====================%
                testt_indx = (indicess == cro);
                trainn_indx = ~testt_indx;
            
                trainn = XX( trainn_indx, :);                         %===================%
                trainn_label = test_label( trainn_indx );                     %      选择训练集    %
                %                                                  %===================%
                testt = XX( testt_indx, :);                       %===================%
                testt_label =test_label(testt_indx);                         %     选择测试集     %
                %results.accur(cros, rn) = knnclassifier(train, test, 5);
                results.test(cro, rn) = knnClassification(T_nClasses,trainn,trainn_label,testt,testt_label);
                iinf = cro + (rn - 1)*cros_num;
                results.rse{iinf} = ela;
            end
        end
        
        
    end
    results.method = type_method;
    save(['Results\', 'res_', type_method,'', name ],  'results');
    clear results;
end

