function [draws] = f_rand_discrete_pdf_1d(pdf, num_draws)
% Returns a randomly drawn bin number for a given discrete (binned) 1-d pdf
% Input
% - pdf: [1,n] or [n,1] array with a discrete (binned) probability density function
%   Note: pdf must be NaN-free and must sum up to 1
% - num_draws: [1,1] desired number of random draws
% Output
% - draws: [n,1] bin indices of the random draws
% Version
% - 2020/04/13 Uwe Ehret: initial version

% check if pdf is NaN-free
if ~isempty(find(isnan(pdf)))
    error('pdf contains NaNs')
end

% check if pdf sums up to almost one
if abs(sum(pdf) - 1) > .00001
    error('Probablities dont sum to 1.')
end

% if pdf is a row vector, convert it to a column vector
% - required for proper concatenation of 'cdf_edges'
dummy = size(pdf);
if dummy(1) > dummy(2)
    pdf = pdf';
end

% construct the cdf(cumulative probability density function) of the pdf
% - cdf(2) contains the sum of pdf(1)+pdf(2), 
%   i.e. its the cumulative probability at the right end of the bin
cdf = cumsum(pdf);

% construct the edges of the cdf by adding a leading zero
% - the bin of cdf(1) is framed by cdf_edges(1) and cdf_edges(2)
cdf_edges = [0 cdf];

% generate 0-1 uniformly distributed random number 
% Note: Results are from open interval (0,1), i.e excluding 0 and 1
rands = rand(num_draws,1);
% rands = [0.25 0.55 0.8 0.85 0.999]';

% get the corresponding bin number in the non-zero cdf
% - histcounts returns the index of the LEFT edge of the bin!
%   this compensates with enumeration shift by adding the 0 left edge
%   for cdf_edges. So the value returned by 'histcounts' is directly
%   the correct bin number corresponding to 'pdf'
[~,~,draws] = histcounts(rands,cdf_edges); 

end

