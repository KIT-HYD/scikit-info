function [h] = f_plot_histogram_2d(hist_in,normalize_each_target_pdf_flag,bincenters)
% Plots a 2-d histogram or pdf
% Note: The plot is arranged such that 
%   - the origin is in the lower left corner
%   - the x-axis shows the predictor values
%   - the y-axis shows the target values
% Input
% - hist_in: [num_bins of dim 1, num_bins of dim 2] matrix, 
%   with 'dim 1' being the target, and 'dim 2' ... 'dim end' being the predictor
% - normalize_each_target_pdf_flag: [1,1] boolean
%   - if 'false', the histogram or pdf will be plotted with values as provided.
%   - if 'true', each target histogram or pdf will be normlaized to sum=1 before plotting
% - bincenters (optional): [1,2] cell array with bin center values
%   - bincenters{1} is an [num_bins of dim 1] array with bin centers of the target
%   - bincenters{2} is an [num_bins of dim 2] array with bin centers of the predictor
%   - if this argument is not provided, the bin number will be plotted instead
% Output
% - h: plot handle
% Version
% - 2020/10/08 Uwe Ehret: added the option to show values normalized for each target pdf
% - 2020/10/01 Uwe Ehret: initial version

% convert 'hist_in' if requested
if normalize_each_target_pdf_flag == true
    
    % loop over all predictor bins
    for p = 1 : size(hist_in,2)
        hist_in(:,p) = hist_in(:,p) ./ sum(hist_in(:,p)); % for each predictor value, normalize the distribution of target values to sum=1
    end
    
end

 if ~exist('bincenters','var')  % bin center values are not provided
                                % --> plot the axes with bin number as tick values
    figure
    h = imagesc(hist_in);
    xlabel('predictor 1 [bin number]');
    ylabel('target [bin number]');
    set(gca,'YDir','normal')  
    colorbar;          
    
 else % bin center values are provided
      % --> plot the axes with given bin center values as tick values
    
      figure
    h = imagesc([bincenters{2}(1) bincenters{2}(end)],[bincenters{1}(1) bincenters{1}(end)],hist_in);
    xlabel('predictor 1 [bin center value]');
    ylabel('target [bin center value]');
    xticks(bincenters{2});
    yticks(bincenters{1});    
    set(gca,'YDir','normal') 
    colorbar;   
    
 end


end

