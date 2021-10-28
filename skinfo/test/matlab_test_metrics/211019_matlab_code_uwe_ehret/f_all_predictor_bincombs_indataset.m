function [ combis ] = f_all_predictor_bincombs_indataset( data, edges )
% returns an array of all unique predictor bin combinations in a dataset
% Input
% - data: [num_data,num_dim] array, where each row is a set of related target
%   (col=1) and predictors (col= 2:end) values
%   Note: 
%   - data must be NaN-free
%   - num_dim must be >=2 (1 target plus at least one predictor)
% - edges: [1,num_dim] cell array, with a [1,num_edges] array of bin edges
%   for each dimension inside
%   Note: For each dimension, the edges must completely cover the entire
%   value range of the respective data
% Output
% - combis: [num_combis,num_dim-1] array with all unique bin number
%   combinations across all predictors in the dataset
% Version
% - 2018/07/24 Uwe Ehret: initial version

% get dimensionality of data set
    [num_data, num_dim] = size(data);

% check input
    % check input data for NaN
    if ~isempty(find(isnan(data)))
        error('input data contain NaN');
    end

    % check for at least one predictor
    if num_dim < 2
        error('need at least two columns in input data');
    end

    % check if input data fall outside the bin edges
    mins = min(data,[],1);   % smallest value in each dimension
    maxs = max(data,[],1);   % largest value in each dimension

    % loop over all dimensions
    for d = 1 : num_dim 
        if mins(d) < edges{d}(1) % smallest value is < leftmost edge
            error('input data < leftmost edge');
        elseif maxs(d) > edges{d}(end)  % largest value is > rightmost edge
            error('input data > rightmost edge');
        end
    end
    
% initialize output variables
% Note: Here for convenience still with the first=target col, will be erased later
combis = NaN(num_data,num_dim);    

% loop over all dimensions (except the first = target)
for d = 2 : num_dim
    % classify the data in each dimension into bins
    [~,~,combis(:,d)] = histcounts(data(:,d),edges{d});
end

% erase the first column (=target)
combis = combis(:,2:end);

% erase all redundant rows
combis = unique(combis,'rows');

end

