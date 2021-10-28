function [pdf_nonzero] = f_NonZeroPDF_anyd(data_histcounts)
% Returns a pdf of an any-dimensional discrete (binned) frequency distribution where all bins have non-zero probabilities
% Method
% - For each bin, its nonzero bin occupation probability is estimated 
%   as the mean of the confidence interval for p_i based on the binominal distribution.
%   These confidence intervals will become narrower the larger the total counts in data_histcounts
% Input
% - data_histcounts: [num_bins of dim 1, num_bins of dim 2, ... , num_bins of dim end]
%   matrix, with counts of occurrency of all possible bin combinations
%   across dimensions
% Note: data_histcounts must be NaN-free
% Output
% - pdf_nonzero: [num_bins of dim 1, num_bins of dim 2, ... , num_bins of dim end]
%   matrix with nonzero bin occupation probability (strictly positive)
% Version
% 2018/11/14 Uwe Ehret, initial version

% check if data_histcounts is NaN-free
if ~isempty(find(isnan(data_histcounts)))
    error('data_histcounts contains NaNs')
end

% reshape data_histcounts to one long 1-d array
data_histcounts_1d = reshape(data_histcounts,[numel(data_histcounts),1]);

% get the total number of counts in the 1-d histogram
num_counts = sum(data_histcounts_1d);

% for each bin, compute the confidence interval of its bin occuptation probability, provided as upper and lower value of 95% confidence interval 
[~,CI] = binofit(data_histcounts_1d,num_counts); 

% the non-zero bin occupation probability is the mean of the confidence interval
pdf_nonzero_1d = mean(CI,2);

% as pdf_nonzero_array = mean(CI,2) does not assure sum(pdf_nonzero)=1, do so by diving with the sum
pdf_nonzero_1d = pdf_nonzero_1d'/sum(pdf_nonzero_1d);    

% reshape pdf_nonzero_1d back to the original dimensions
pdf_nonzero = reshape(pdf_nonzero_1d,size(data_histcounts));     

end