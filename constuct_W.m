function S = constuct_W(M, K, epsilon)
    % M: d * N   (features_dim * data amount)
    [idx, dist]= knnsearch(M', M','K', K+1, 'Distance', 'euclidean');
    idx = idx(:, 2:end); 
    dist = dist(:, 2:end);
    
    N = size(M, 2);
    S = zeros(N, N);
    for i=1:N
        S(i, idx(i,:)) = exp(-dist(i,:).*dist(i,:)/epsilon) + 10^-8; % add small noise to avoid singularlity
    end
    S = S + S';         %   Symmetric the graph
    
    %   Normalize such that row sum is 1;
    S = S./kron(ones(1, N),(sum(S'))');
end



