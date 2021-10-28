clear all
close all
clc

%% load data
% test data created with Python and numpy:
% np.random.seed(42)
% n = np.random.normal(40, 2.5, 1000)
% np.random.seed(2409)
% g = np.random.gamma(10, 7, 1000)

% histcounts
% bins = np.histogram_bin_edges(n, bins='fd')
% count = np.histogram(n, bins=bins)[0]
n_histcounts = csvread("test_data/n/test_data_n_histcounts_fd.csv");
g_histcounts = csvread("test_data/g/test_data_g_histcounts_fd.csv");

% pdf: same bins for n & g
% np.histogram_bin_edges([n, g], bins='fd')
% hist_n = np.histogram(n, bins=bins)[0]
% hist_g = np.histogram(g, bins=bins)[0]
% p_n = (hist_n / np.sum(hist_n)) + 1e-15
% p_g = (hist_g / np.sum(hist_g)) + 1e-15
n_pdf_fd_same_bins = csvread("test_data/n/test_data_n_pdf_same_bins_fd.csv");
g_pdf_fd_same_bins = csvread("test_data/g/test_data_g_pdf_same_bins_fd.csv");

% joint hist_counts
% bins_joint = [bins_n, bins_g]
% joint_counts = np.histogram2d(n, g, bins_joint)[0]
ng_histcounts_fd = csvread("test_data/test_data_n_g_joint_counts_fd.csv");

%% Entropy
%in: einfach array [...]
entropy_n = f_entropy_anyd(n_histcounts);
entropy_g = f_entropy_anyd(g_histcounts);

%% Conditional Entropy
conditional_entropy_ng = f_conditionalentropy_anyd(ng_histcounts_fd, f_all_predictor_bincombs([28,27]));
conditional_entropy_gn = f_conditionalentropy_anyd(ng_histcounts_fd.', f_all_predictor_bincombs([27,28]));

%% Mutual Information
mutual_info_ng = f_entropy_anyd(n_histcounts) - f_conditionalentropy_anyd(ng_histcounts_fd, f_all_predictor_bincombs([28,27]));

%% Cross Entropy
%in: einfach zwei arrays [...], [...]
cross_entropy_ng = f_crossentropy(n_pdf_fd_same_bins, g_pdf_fd_same_bins);
cross_entropy_gn = f_crossentropy(g_pdf_fd_same_bins, n_pdf_fd_same_bins);

%% Joint Entropy
%in: beide pdfs -> berechnet joint entropy
joint_entropy_ng = f_entropy_anyd(ng_histcounts_fd);

%% Kullback-Leibler
%in: einfach zwei arrays [...], [...]
kld_ng = f_kld_anyd(n_pdf_fd_same_bins, g_pdf_fd_same_bins);
kld_gn = f_kld_anyd(g_pdf_fd_same_bins, n_pdf_fd_same_bins);

%% Jensen-Shannon
%in: einfach zwei arrays [...], [...]
jsd_ng = f_jsd_anyd(n_pdf_fd_same_bins, g_pdf_fd_same_bins);


%% Save results
results = table(entropy_g, entropy_n, conditional_entropy_ng, ...
    conditional_entropy_gn, mutual_info_ng, cross_entropy_gn, ...
    cross_entropy_ng, joint_entropy_ng, kld_gn, kld_ng, jsd_ng)

writetable(results, "results_matlab.csv")
