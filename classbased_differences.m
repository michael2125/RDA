function [Cmean, Allmean, ccmean_m_xmeans, xxi_m_cmeans, nis] = classbased_differences(x_train, y_train)
% [cmean_m_xmean, xi_m_cmean, nis] = classbased_differences(Xs, classes)
%
% input:
% Xs: cell array of (multi-dimensional) matrices
% classes: classes of matrices
%
% output:
% cmean_m_xmean: class means minus overall mean
% xi_m_cmean: observations minus corresponding class mean
x_train_cell = mat_to_cell( x_train.data );
Ksize = size(x_train);
KKsize = Ksize(1:end-1);
Xs = x_train_cell;
classes = y_train;
% =========================================================================
nsamples = length(Xs);
nclasses = length(unique(classes));
Cmean = tenzeros(Ksize);
Allmean = tenzeros(Ksize);



Xsum = Xs{1};
for isample = 2:nsamples
    Xsum = Xsum + Xs{isample};
end
Xmean = Xsum/nsamples;
% ======================= Construct Allmean tensor ========================

for isample = 1:nsamples

            switch length(Ksize)
            case 3
            Allmean(:,:,isample) = Xmean;
            case 4
            Allmean(:,:,:,isample) = Xmean;
            case 5
            Allmean(:,:,:,:,isample) = Xmean;
            case 6
            Allmean(:,:,:,:,:,isample) = Xmean;
            otherwise
            error('Ksize is not between 3 and 6.');
            end
end
%nmodes = length(size(Xs{1}));
%catXs=cat(nmodes+1,Xs{:});
%Xmean = mean(catXs, nmodes+1); 

Xmeansclasses = cell(1, nclasses);
Xsumsclasses = cell(1, nclasses);
nis = NaN(1, nclasses);
for iclass = 1:nclasses
    inds = find(classes==iclass);
    Xsumsclasses{iclass} = Xs{inds(1)};
    for iind = 2:length(inds)
        Xsumsclasses{iclass} = Xsumsclasses{iclass} + Xs{inds(iind)};
    end
    nis(iclass) = length(inds);
    Xmeansclasses{iclass} = Xsumsclasses{iclass}/nis(iclass);
    
% ============================ Construct Cmean tensor =====================
  for i = 1:length(inds)
          switch length(Ksize)
            case 3
            Cmean(:,:,inds(i)) = Xmeansclasses{iclass};
            case 4
            Cmean(:,:,:,inds(i)) = Xmeansclasses{iclass};
            case 5
            Cmean(:,:,:,:,inds(i)) = Xmeansclasses{iclass};
            case 6
            Cmean(:,:,:,:,:,inds(i)) = Xmeansclasses{iclass};
            otherwise
            disp('Ksize is not between 3 and 6.');
          end
  end
end
% ======================== Construct Sw norm ==============================
xi_m_cmeans = cell(1, nsamples);
xxi_m_cmeans = tenzeros([KKsize nsamples]);
for isample = 1:nsamples
    xi_m_cmeans{isample} = Xs{isample}-Xmeansclasses{classes(isample)};
          switch length(Ksize)
            case 3
            xxi_m_cmeans(:,:,isample) = xi_m_cmeans{isample};
            case 4
            xxi_m_cmeans(:,:,:,isample) = xi_m_cmeans{isample};
            case 5
            xxi_m_cmeans(:,:,:,:,isample) = xi_m_cmeans{isample};
            case 6
            xxi_m_cmeans(:,:,:,:,:,isample) = xi_m_cmeans{isample};
            otherwise
            disp('Ksize is not between 3 and 6.');
          end
end

% ======================== Construct Sb norm ==============================
cmean_m_xmeans = cell(1, nclasses);
ccmean_m_xmeans = tenzeros([KKsize nclasses]);
for iclass = 1:nclasses
    cmean_m_xmeans{iclass} = Xmeansclasses{iclass}-Xmean;
         switch length(Ksize)
            case 3
            ccmean_m_xmeans(:,:,iclass) = cmean_m_xmeans{iclass};
            case 4
            ccmean_m_xmeans(:,:,:,iclass) = cmean_m_xmeans{iclass};
            case 5
            ccmean_m_xmeans(:,:,:,:,iclass) = cmean_m_xmeans{iclass};
            case 6
            ccmean_m_xmeans(:,:,:,:,:,iclass) = cmean_m_xmeans{iclass};
            otherwise
            disp('Ksize is not between 3 and 6.');
         end
end

end


