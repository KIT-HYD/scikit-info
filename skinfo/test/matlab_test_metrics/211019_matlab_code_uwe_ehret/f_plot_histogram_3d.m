function [h] = f_plot_histogram_3d(hist_in,normalize_each_target_pdf_flag,target_slices,p1_slices,p2_slices,bincenters)
% plots a 3-d histogram or pdf
% Note: The plot is arranged such that 
%   - the origin is in the lower left corner
%   - the x-axis shows the predictor 1 values
%   - the y-axis shows the predictor 2 values
%   - the z-axis shows the target values
% Input
% - hist_in: [num_bins of dim 1, num_bins of dim 2, num_bins of dim 3] matrix, 
%   with 'dim 1' being the target, and 'dim 2' being predictor 1, and 'dim 3' being predictor 2
% - normalize_each_target_pdf_flag: [1,1] boolean
%   - if 'false', the histogram or pdf will be plotted with values as provided.
%   - if 'true', each target histogram or pdf will be normlaized to sum=1 before plotting
% - target_slices: [ ] scalar or vector. At the specified locations, slice planes orthogonal to 
%   the target (z)-axis will be drawn. Pass '[]' if no plane is to be drawn
% - p1_slices: Like 'target_slices', but for slices orthogonal to the predictor 1 (x)-axis
% - p2_slices: Like 'target_slices', but for slices orthogonal to the predictor 2 (y)-axis
% - bincenters (optional): [1,3] cell array with bin center values
%   - bincenters{1} is an [num_bins of dim 1] array with bin centers of the target
%   - bincenters{2} is an [num_bins of dim 2] array with bin centers of predictor 1
%   - bincenters{3} is an [num_bins of dim 3] array with bin centers of predictor 2
% Note: if 'bincenters' are not provided
% - the bin number will be plotted instead
% - the slice positions have to be given in units of bin numbers
% - if bincenters are provided, the slice positions have to be given in
%   the units of the binceners
% Output
% - h: plot handle
% Version
% - 2020/10/08 Uwe Ehret: added the option to show values normalized for each target pdf
% - 2020/10/01 Uwe Ehret: initial version

% convert 'hist_in' if requested
if normalize_each_target_pdf_flag == true
    
    % loop over all predictor bins
    for p1 = 1 : size(hist_in,2)        % predictor 1
        for p2 = 1 : size(hist_in,3)    % predictor 2
            hist_in(:,p1,p2) = hist_in(:,p1,p2) ./ sum(hist_in(:,p1,p2)); % for each predictor value pair, normalize the distribution of target values to sum=1
        end
    end
    
end

% rearrange the matrix for correct assignment of predictor 1 to x-axis, predictor 2 to y-axis, and target to z-axis 
hist_plot = permute(hist_in,[3 2 1]);

 if ~exist('bincenters','var') % plot the axes with bin number as tick values
    figure
    h = slice(hist_plot,p1_slices,p2_slices,target_slices);
    xlabel('predictor 1 [bin number]');
    ylabel('predictor 2 [bin number]');
    zlabel('target [bin number]');
    % set(gca,'YDir','normal'); % not required, just here as a reminder that this option exists 
    colorbar; 
       
 else % plot the axes with given bin center values as tick values
     
    % create the coordinate data
    [X,Y,Z] = meshgrid(bincenters{2},bincenters{3},bincenters{1});
   
    % plot
    figure       
    h = slice(X,Y,Z,hist_plot,p1_slices,p2_slices,target_slices);   
    xlabel('predictor 1 [bin center value]');
    ylabel('predictor 2 [bin center value]');
    zlabel('target [bin center value]');
    % set(gca,'YDir','normal'); % not required, just here as a reminder that this option exists 
    xticks(bincenters{2});
    yticks(bincenters{3});  
    zticks(bincenters{1});  
    colorbar;            

 end

% adjust the looks (make empty bins transparent)    

    % Loop over all slices (every slice has its own handle)
    for n = 1:length(h) 
        set(h(n), 'alphadata', get(h(n), 'cdata'), 'FaceAlpha', 'flat','EdgeAlpha', 'flat' );   % set alphadata to cdata ( = colourdata) 
    end

    % adjust alphamap so 0 values are completely transparent and the other
    % values are more opaque than in default
    a = alphamap('rampup',40);
    a(11:35) = 0.7;     % adjust alphamap (can also be printed as lineplot) 
    alphamap(a); % only changes alphamap of current figure    

end

