function [ HPQ ] =f_crossentropy(pdf,pdf_star)
% computes the Crossentropy between two distributions. 
% Note 
% - it is non-symmetrical!
% - HPQ(pdf||pdf_star) = H(pdf) + DKL(pdf||pdf_star)
% Input
% - pdf: [n,1] or [1,n] vector of probabilities representing the reference distribution (the 'truth')
% - pdf_star: [n,1] or [1,n] vector of probabilities representing the other distribution (the 'estimate')
%   Note
%   - pdf and pdf_star must have the same dimension
%   - The elements of pdf and pdf_star must each sum to 1 +/- .00001.
%   - All elements of pdf_star must be non-zero 
%   - In pdf, zero values are allowed
% Output
% - HPQ: [1,1] Crossentropy in [bit]
% Version
% - 2017/10/26 Uwe Ehret: intial version

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
if abs(sum(pdf) - 1) > .00001
    error('Probablities in pdf dont sum to 1.')
end

% check probabilities in 'pdf_star' sum to 1
if abs(sum(pdf_star) - 1) > .00001
    error('Probablities in pdf_star dont sum to 1.')
end

% check for zero values in pdf_star where pdf is non-zero
if ~isempty(intersect(find(pdf_star == 0), find(pdf ~= 0)))
    error('there are zero probabilities in pdf_star where pdf is non-zero');
end

% HPQ(pdf||pdf_star) = H(pdf) + DKL(pdf||pdf_star)
HPQ = f_entropy_anyd(pdf) + f_kld_anyd(pdf,pdf_star);

end
