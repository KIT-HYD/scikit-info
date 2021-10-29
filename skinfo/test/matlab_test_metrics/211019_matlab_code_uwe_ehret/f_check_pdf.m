function [  ] = f_check_pdf(pdf)
% Checks a pdf for sum = 1+/- 0.00001. Throws an error if not
% Input
% - pdf: [1,n] or [n,1] array with a probability density function
%   Note: pdf must be NaN-free
% Output
% - none (no error)
% Version
% - 2017/10/25 Uwe Ehret: initial version

% check if pdf is NaN-free
if ~isempty(find(isnan(pdf)))
    error('pdf contains NaNs')
end

% check if pdf sums up to almost one
if abs(sum(pdf) - 1) > .00001
    error('Probablities dont sum to 1.')
end

end

