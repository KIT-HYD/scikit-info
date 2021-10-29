% 2017/10/25 Uwe Ehret
% % collection of tests for information-based functions

clear all
close all
clc

% check f_entropy_anyd

    in = [50 25 25];
    out = f_entropy_anyd(in)

    in = [50 25 25; 0 0 0];
    out = f_entropy_anyd(in)

    in = [1 0 0;0 1 0];
    out = f_entropy_anyd(in)
    out = f_entropy_anyd(in)

    in = zeros(3,3,3);
    in(1,1,1) = 1;
    in(3,3,3) = 1;
    out = f_entropy_anyd(in)

    in_1 = NaN(2,3,2);
    in_1(1,1,1) = 1;
    in_1(1,2,1) = 3;
    in_1(1,3,1) = 0;
    in_1(2,1,1) = 3;
    in_1(2,2,1) = 3;
    in_1(2,3,1) = 0;
    in_1(1,1,2) = 5;
    in_1(1,2,2) = 3;
    in_1(1,3,2) = 8;
    in_1(2,1,2) = 3;
    in_1(2,2,2) = 3;
    in_1(2,3,2) = 0;
    out = f_entropy_anyd(in_1)

% f_rand_discrete_pdf_1d
    pdf = [0 0.5 0.3 0 0.2 0];
    out = f_rand_discrete_pdf_1d(pdf,1000);

% f_jsd_anyd
    pdf_1 = [0 1 0 0];
    pdf_2 = [1 0 0 0];
    JSD = f_jsd_anyd(pdf_1, pdf_2)

    pdf_1 = [1 0 0 0];
    pdf_2 = [1 0 0 0];
    JSD = f_jsd_anyd(pdf_1, pdf_2)

    pdf_1 =[0.26 0.24 0.25 0.25];
    pdf_2 = [0.25 0.25 0.25 0.25];
    JSD = f_jsd_anyd(pdf_1, pdf_2)

    pdf_1 = [0 0 0 0.5; 0 0 0 0.5];
    pdf_2 = [0 0 0 0; 0 0 1 0];
    JSD = f_jsd_anyd(pdf_1, pdf_2)

% f_all_predictor_bincombs
    in = [2,5,3];
    out = f_all_predictor_bincombs(in);

% test f_check_pdf
    in = [0 0 1]';
    f_check_pdf(in);

% check f_conditionalentropy_anyd
    in_1 = NaN(2,3);
    in_1(1,1) = 1;
    in_1(1,2) = 3;
    in_1(1,3) = 0;
    in_1(2,1) = 3;
    in_1(2,2) = 3;
    in_1(2,3) = 0;
    in_2 = f_all_predictor_bincombs([2,3]);
    [out_1,out_2] = f_conditionalentropy_anyd(in_1,in_2);

% check_histcounts_anyd
    A = NaN(10,3);
    A(:,1) = [0 1 0 1 1 0 1 1 1 0];
    A(:,2) = [1 1 3 1 1 2 2 2 2 2];
    A(:,3) = [10 10 10 20 20 20 20 40 40 40];
    A_edges = cell(1,3);
    A_edges{1} = [-0.5 0.5 1.5];
    A_edges{2} = [0.5 1.5 2.5 3.5];
    A_edges{3} = [5 15 25 35 45];
    [A_binned, A_histcounts] = f_histcounts_anyd(A, A_edges);

% check f_kld_anyd
    in_1 = [0 1];
    in_2 = [0.5 0.5];
    out_1 = f_kld_anyd(in_1,in_2);

% check f_NonZeroPDF_anyd
    in_1 = NaN(3,4,2);
    in_1(:,:,1) = [1 1 1 100 ; 1 1 1 1 ; 1 1 1 1];
    in_1(:,:,2) = [0 0 0 0 ; 0 0 0 0 ; 100 0 0 0];
    out_2 = f_NonZeroPDF_anyd(in_1);

% check f_sample_data
    A = NaN(10,3);
    A(:,1) = [0 1 0 1 1 0 1 1 1 0];
    A(:,2) = [1 1 3 1 1 2 2 2 2 2];
    A(:,3) = [10 10 10 20 20 20 20 40 40 40];
    out_1 = f_sample_data(A,2,'continuous');
    out_2 = f_sample_data(A,2,'continuous');

% check f_crossentropy
    in_1 = [0 1];
    in_2 = [0.5 0.5];
    out_1 = f_crossentropy(in_1,in_2);

% check f_conditional_pdf
    A = NaN(10,3);
    A(:,1) = [0 1 0 1 1 0 1 1 1 0];
    A(:,2) = [1 1 3 1 1 2 2 2 2 2];
    A(:,3) = [10 10 10 20 20 20 20 40 40 40];
    A_edges = cell(1,2);
    A_edges{1} = [-0.5 0.5 1.5];
    A_edges{2} = [0.5 1.5 2.5 3.5];
    A_edges{3} = [5 15 25 35 45];
    [A_binned, A_histcounts] = f_histcounts_anyd(A, A_edges);
    predictor_set = [3, 4];
    cpdf = f_conditional_histogram(predictor_set,A_histcounts);
