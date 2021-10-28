function [H] = f_entropy_anyd(data_histcounts)
% Computes the entropy of an 1-to-any-dimensional discrete (binned) frequency distribution
% This is also the joint entropy of all the variables that together build this any-dimensional space
% Properties
% - Nonnegativity: H >= 0
% - Greater_equal individual entropies: H(X1,...Xn) >= max[H(X1),..H(Xn)]
% - Smaller_equal sum of individual entropies: H(X1,...Xn) <= H(X1) + ... + H(xn)
% Input
% - data_histcounts: [num_bins of dim 1, num_bins of dim 2, ... , num_bins of dim end]
%   matrix, with bin counts of all possible bin combinations across dimensions
% Note: data_histcounts must be NaN-free
% Output
% - H: [1,1] entropy in [bit]
% Version
% - 2020/11/26 Uwe Ehret: Added some more info in the header
% - 2018/07/13 Uwe Ehret: initial version

% check if 'data_histcounts' is NaN-free
if ~isempty(find(isnan(data_histcounts)))
    error('data_histcounts contains NaNs')
end

% reshape 'data_histcounts' to one long array
data_histcounts_array = reshape(data_histcounts,[numel(data_histcounts),1]);

% normalize data_histcounts_array to a pdf
pdf = data_histcounts_array / sum(data_histcounts_array);

% compute the entropy of the 1-d pdf
H = f_entropy(pdf);
    
end

