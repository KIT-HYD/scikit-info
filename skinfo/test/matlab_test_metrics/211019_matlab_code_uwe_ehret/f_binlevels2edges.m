function [edges, centers] = f_binlevels2edges(indx_binlevels, indx_datatypes, binlevels, datatypes)
% For an array of datatypes, returns bin edges for a desired binlevel
% number 
% Input
% - indx_binlevels: [1,num_dim] array with desired binlevel for each
%       dimension. The binlevel is used as index in binlevels to retrieve
%       the desired number of bins
% - indx_datatypes: [1,num_dim] :double. Datatype index for each dimension.%       
% - binlevels: [m,1] :double. Specifies the number of equal-sized bins 
%       the datatype range [min, max] should be subdivided into
% - datatypes [n,1] :struct. Specifies the datatype
% Output
% - edges: [1,num_dim] cell array, with arrays of bin edges for each dimension
% - centers: [1,num_dim] cell array, with arrays of bin centers for each dimension
% Version
% - 2018/07/13 Uwe Ehret: initial version

% get dimensionality (= number of edge-arrays to create)
num_dim = size(indx_binlevels,2);

% initialize output variables
edges = cell(1,num_dim);
centers = cell(1,num_dim);

% loop over all dimensions 
for i = 1 : num_dim

    % get parameters for binning
    min = datatypes(indx_datatypes(i)).min;     % min value for given datatype
    max = datatypes(indx_datatypes(i)).max;     % max value for given datatype
    numbins = binlevels(indx_binlevels(i));     % number of bins for desired binlevel
    
    % create the bins and centers
    var_edges = linspace(min, max, numbins+1);          % create the bin edges
    var_centers = var_edges(1:end-1) + diff(var_edges) / 2;     % create the bin centers 
    
    % write edges and centers to output
    edges{i} = var_edges;
    centers{i} = var_centers;

end

end

