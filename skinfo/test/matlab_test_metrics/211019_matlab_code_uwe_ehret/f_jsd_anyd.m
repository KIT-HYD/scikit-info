function [JSD] = f_jsd_anyd(pdf_1,pdf_2)
% returns Jensen-Shannon divergence between two discrete distributions of any dimensionality. 
% Note 
% - it is symmetrical
% - it is a true metric
% Input
% - pdf_1, pdf_2: x-dimensional discrete (binned) pdf's to be compared, each normalized to sum=1
%   Note
%   - pdf_1 and pdf_2 must have the same dimensionsionalty (number of dimensions and number of bins along each dimension)
%   - The elements of pdf_1 and pdf_2 must each sum to 1 +/- .00001.
%   - In pdf_1 and pdf_2, zero values are allowed
% Output
% - JSD: [1,1] Jensen-Shannon-divergence in [bit]
% Version
% - 2019/12/08 Uwe Ehret: intial version

% check input
    % check if there are NaNs in 'pdf_1'
    if ~isempty(find(isnan(pdf_1)))
        JSD = NaN;
        return;
    end

    % check if there are NaNs in 'pdf_2'
    if ~isempty(find(isnan(pdf_2)))
        JSD = NaN;
        return;
    end

    % check for equal input dimensions
    if ~isequal(size(pdf_1),size(pdf_2))
        error('All inputs must have same dimension.')
    end

    % check probabilities in 'pdf_1' sum to 1
    if abs(sum(pdf_1(:)) - 1) > .00001
        error('Probablities in pdf_1 dont sum to 1.')
    end

    % check probabilities in 'pdf_2' sum to 1
    if abs(sum(pdf_2(:)) - 1) > .00001
        error('Probablities in pdf_2 dont sum to 1.')
    end

% create a mean distribution by AND-combination of pdf_1 and pdf_2
pdf_mean = (pdf_1 + pdf_2)/2;

% calculate JSD as the mean of DKL(pdf_1||pdf_mean) and DKL(pdf_2||pdf_mean)
JSD = (f_kld_anyd(pdf_1,pdf_mean) + f_kld_anyd(pdf_2,pdf_mean))/2;

end

