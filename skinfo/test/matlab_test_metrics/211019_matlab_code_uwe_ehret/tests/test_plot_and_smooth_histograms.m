% 2020/10/01 Uwe Ehret
% Tests functions to plot and smooth 2-d and 3-d histograms
% overview on smoothing functions
% - smooth: works only for 1-d data
% - smooth3: works only for 3-d data
% - imfilter:works for ANY-d data --> seems possible
% - smoothn: works for ANY-d data! --> Use this

clear all
close all
clc

%% create histograms for plotting

    binlevels = [0 1 2 4 8 16 32 64 128 256 512 1024 2048]';
    datatypes(1) = struct('type','p1','unit','m','min',1,'max',2);
    datatypes(2) = struct('type','p2','unit','m','min',5,'max',10);    
    datatypes(3) = struct('type','t','unit','N','min',5,'max',20);

    [pred_1_edges, pred_1_centers] = f_binlevels2edges(5,1, binlevels, datatypes);
    [pred_2_edges, pred_2_centers] = f_binlevels2edges(6,2, binlevels, datatypes);
    [target_edges, target_centers] = f_binlevels2edges(7,3,binlevels,datatypes);

    p1_vals = linspace(1,2,100)';
    p2_vals = linspace(5,10,100)';
    all_vals = [];
    for x = 1 : length(p1_vals)
        for y = 1 : length(p2_vals)
            bla = [p1_vals(x)*p2_vals(y) p1_vals(x) p2_vals(y)];
            all_vals = [all_vals; bla];
        end
    end

    % 2-d histogram
    all_edges = [target_edges pred_1_edges];
    all_vals_2d = all_vals(:,1:2);
    [~, hist_2d_sharp] = f_histcounts_anyd(all_vals_2d, all_edges);  

    % 3-d histogram
    all_edges = [target_edges pred_1_edges pred_2_edges];
    all_vals_3d = all_vals;
    [~, hist_3d_sharp] = f_histcounts_anyd(all_vals_3d, all_edges);  

%% smooth histograms
       
    % 2-d histogram
    hist_2d_smooth = smoothn(hist_2d_sharp,0.5);

    % 3-d histogram
    hist_3d_smooth = smoothn(hist_3d_sharp,10);

%% plot histograms

    % 2-d plots
    all_centers = [target_centers pred_1_centers];
    h = f_plot_histogram_2d(hist_2d_sharp,false);
    h = f_plot_histogram_2d(hist_2d_smooth,false);
    h = f_plot_histogram_2d(hist_2d_smooth,true);
    h = f_plot_histogram_2d(hist_2d_smooth,false,all_centers);
    h = f_plot_histogram_2d(hist_2d_smooth,true,all_centers);    

    % 3-d plots
    target_slices_bin = [];
    p1_slices_bin = [7];
    p2_slices_bin = [5 13];
    h = f_plot_histogram_3d(hist_3d_smooth,false,target_slices_bin,p1_slices_bin,p2_slices_bin);
    h = f_plot_histogram_3d(hist_3d_smooth,true,target_slices_bin,p1_slices_bin,p2_slices_bin);

    
    target_slices = [];
    p1_slices = [1.8125];
    p2_slices = [6.4063 8.9063];
    all_centers = [target_centers pred_1_centers pred_2_centers];
    h = f_plot_histogram_3d(hist_3d_sharp,false,target_slices,p1_slices,p2_slices,all_centers);
    h = f_plot_histogram_3d(hist_3d_sharp,true,target_slices,p1_slices,p2_slices,all_centers);
    h = f_plot_histogram_3d(hist_3d_smooth,false,target_slices,p1_slices,p2_slices,all_centers);
    h = f_plot_histogram_3d(hist_3d_smooth,true,target_slices,p1_slices,p2_slices,all_centers);

