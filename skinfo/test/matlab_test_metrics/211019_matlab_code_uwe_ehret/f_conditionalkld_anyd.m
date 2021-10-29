function [ KLDc ] = f_conditionalkld_anyd( pdf, pdf_star, p_combis )
% returns the conditional Kullback-Leibler divergence (KLD of a single target variable)
% given any number of predictors >= 1, i.e. dimensions of the pdfs must be at least 2 (1 target, 1 predictor)
% Note
% - the target is given in the first dimension
%   - pdf and pdf_star must have the same dimensionsionalty (number of dimensions and number of bins along each dimension)
%   - The elements of pdf and pdf_star must each sum to 1 +/- .00001.
%   - In pdf, zero values are allowed (divergence in this case will be 0)
%   - In pdf_star 
%     - zero values are allowed where pdf is also zero (divergence in this case will be 0)
%     - zero values where pdf is NOT zero throw an error (divergence in this case would be infinite)
% Input
% - pdf: x-dimensional discrete (binned) pdf normalized to sum=1 representing the reference distribution (the 'truth')
% - pdf_star: x-dimensional discrete (binned) pdf normalized to sum=1  representing the other distribution (the 'estimate')
% - p_combis: [num_p_combis, num_dim -1] array with all possible bin number combinations (rows) of all predictors (columns) in pdf
%   Note this excludes the first dimension in pdf, as this is the target
% Output
% - KLDc: [1,1] Conditional Kullback-Leibler-divergence in [bit]
% Version
% - 2018/07/23: Uwe Ehret, initial version

% check input
    % check if there are NaNs in 'pdf'
    if ~isempty(find(isnan(pdf)))
        KLD = NaN;
        return;
    end

    % check if there are NaNs in 'pdf_star'
    if ~isempty(find(isnan(pdf_star)))
        KLD = NaN;
        return;
    end

    % check for equal input dimensions
    if ~isequal(size(pdf),size(pdf_star))
        error('All inputs must have same dimension.')
    end

    % check probabilities in 'pdf' sum to 1
    if abs(sum(pdf(:)) - 1) > .00001
        error('Probablities in pdf dont sum to 1.')
    end

    % check probabilities in 'pdf_star' sum to 1
    if abs(sum(pdf_star(:)) - 1) > .00001
        error('Probablities in pdf_star dont sum to 1.')
    end

    % check for zero values in pdf_star where pdf is non-zero
    if ~isempty(intersect(find(pdf_star == 0), find(pdf ~= 0)))
        error('there are zero probabilities in pdf_star where pdf is non-zero');
    end

    % check if p_combis is NaN-free
    if ~isempty(find(isnan(p_combis)))
        error('p_combis contains NaNs')
    end
    
% initialize the output variable
KLDc = 0;
    
% get dimensions
num_p_combis = size(p_combis,1); % number of possible predictor combinations

% initialize arrays for all conditional distributions
KLDc_temp = NaN(num_p_combis,1); % target KLD for each particular predictor combination
num_temp = NaN(num_p_combis,1); % number of target values for each particular predictor combination (marginal predictor frequency)

% loop over all particular predictor combinations
for c = 1 : num_p_combis
    
    % get the current predictor combination
    predictor_vals = p_combis(c,:);
    
    % the reference (truth)
    
        % with the current predictor combination, extract the conditional target pdf
        target_pdf_ref = f_conditional_histogram(predictor_vals,pdf);
        
        % find the marginal frequency of the particular predictor combination
        num_temp(c) = sum(target_pdf_ref); 
    
        % get conditional target pdf by dividing its histogram with its frequency
        % Note: If the marginal frequency = 0, this means division by 0, which leads to a pdf filled with NaNs
        target_pdf_ref = target_pdf_ref / sum(target_pdf_ref);
    
    % the estimate (model)
    
        % with the current predictor combination, extract the conditional target pdf
        target_pdf_star = f_conditional_histogram(predictor_vals,pdf_star);

        % get conditional target pdf by dividing its histogram with its frequency
        % Note: If the marginal frequency = 0, this means division by 0, which leads to a pdf filled with NaNs
        target_pdf_star = target_pdf_star / sum(target_pdf_star);    
        
    % compute DKL
    % Note: Will be NaN if either of the two pdfs is filled with NaNs
    KLDc_temp(c) = f_kld_anyd(target_pdf_ref,target_pdf_star); 
        
end

% convert the reference marginal frequencies to marginal probabilities 
num_temp = num_temp / sum(num_temp);

% compute total conditional KLD as expected values of all particular KLDs
KLDc = nansum(num_temp .* KLDc_temp); 
% Note this requires 'nansum' instead of 'sum', as KLDc_temp = NaN for all
% estimate histograms with no data at all.

end

