function [combis] = f_all_bincombs(num_bins)
% Creates an array of all possible bin combinations
% Input
% - num_bins: [1,n] array, where for each dimension the number of bins is given
% Output
% - combis: [num_combis,num_dim] array with all possible bin number combinations
% Version
% - 2017/07/15 Uwe Ehret: initial version

% number of dimensions of the matrix
num_dim = size(num_bins,2); 

% initialze cell array with all possible bin numbers for each dimension
mycell = cell(1,num_dim);

% loop over all dimensions of the matrix
for d = 1 : num_dim
    % write an array with all possible bin numbers
    mycell{d} = (1:num_bins(d)); 
end

% create all possible combinations of predictor bin numbers
combis = allcomb_singleinput(mycell); 

end

