function [ edges ] = f_create_binning( center_left, center_right, binwidth )
% creates an array of bin edges with uniform bin width
% Input
% - center_left: [1,1] center of the leftmost bin
% - center_right: [1,1] center of the rightmost bin
% - binwidth: [1,1] uniform binwidth for all bins
% Output
% - edges: [numbins+1,1] array of bin edges
% Version
% - 2021/06/14 Uwe Ehret: Changed input check from 'mod(numbins)' to 'mod(numbins,1)'
% - 2018/10/25 Uwe Ehret: initial version

% compute bin edges
mini = center_left-0.5*binwidth;   % leftmost bin edge
maxi = center_right+0.5*binwidth;   % rightmost bin edge
numbins = 1+(center_right-center_left)/binwidth;   % number of bins

% check input
if mod(numbins,1) ~= 0
    error('non-integer number of bins for given binwidth and centers!');
end
    
% compute result    
edges = linspace(mini,maxi,numbins+1);

end

