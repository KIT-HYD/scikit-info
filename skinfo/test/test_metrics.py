import pytest

import skinfo.metrics as skinfo

import numpy as np
import pandas as pd


def entropy_test(n, g, matlab_res):
    # test entropy with data n & g
    assert skinfo.entropy(n, bins='fd') == pytest.approx(matlab_res['entropy_n'][0])
    assert skinfo.entropy(g, bins='fd') == pytest.approx(matlab_res['entropy_g'][0])
    
    # test coinflip: entropy of a coin flip equals 1
    assert skinfo.entropy(np.asarray([0, 1]), bins=2) == pytest.approx(1)
    # test parameter use_probs in the coin flip case
    assert skinfo.entropy(np.asarray([0.5, 0.5]), bins=2, use_probs=True) == pytest.approx(1)
    
    # test normalize (hard to test, but can never be greater than 1)
    assert skinfo.entropy(n, bins='fd', normalize=True) <= 1
    assert skinfo.entropy(g, bins='fd', normalize=True) <= 1
    
    return True


def conditional_entropy_test(n, g, matlab_res):
    # test conditional entropy with data n & g (not symmetric)
    assert skinfo.conditional_entropy(n, g, bins='fd') == pytest.approx(matlab_res['conditional_entropy_ng'][0])
    assert skinfo.conditional_entropy(g, n, bins='fd') == pytest.approx(matlab_res['conditional_entropy_gn'][0])
    
    # test normalize (hard to test, but can never be greater than 1)
    assert skinfo.conditional_entropy(n, g, bins='fd', normalize=True) <= 1
    assert skinfo.conditional_entropy(g, n, bins='fd', normalize=True) <= 1
    
    return True  


def mutual_information_test(n, g, matlab_res):
    # test mutual information with data n & g (symmetric)
    assert skinfo.mutual_information(n, g, bins='fd') == pytest.approx(matlab_res['mutual_info_ng'][0])
    assert skinfo.mutual_information(g, n, bins='fd') == pytest.approx(matlab_res['mutual_info_ng'][0])
    
    # test normalize (hard to test, but can never be greater than 1)
    assert skinfo.mutual_information(n, g, bins='fd', normalize=True) <= 1
        
    return True 


def cross_entropy_test(n, g, matlab_res):
    # test cross entropy with data n & g (symmetric)
    assert skinfo.cross_entropy(n, g, bins='fd') == pytest.approx(matlab_res['cross_entropy_ng'][0])
    assert skinfo.cross_entropy(g, n, bins='fd') == pytest.approx(matlab_res['cross_entropy_gn'][0])
    
    # calculate probabilities to test parameter use_probs
    # get the bins
    bins = np.histogram_bin_edges([n, g], 'fd')
    # calculate unconditioned histograms
    hist_n = np.histogram(n, bins=bins)[0]
    hist_g = np.histogram(g, bins=bins)[0]
    #calculate probabilities
    pn = (hist_n / np.sum(hist_n))
    pg = (hist_g / np.sum(hist_g))
    
    # test parameter use_probs
    assert skinfo.cross_entropy(pn, pg, bins='fd', use_probs=True) == skinfo.cross_entropy(n, g, bins='fd')
        
    return True 


def joint_entropy_test(n, g, matlab_res):
    # test joint entropy with data n & g (symmetric)
    assert skinfo.joint_entropy(n, g, bins='fd') == pytest.approx(matlab_res['joint_entropy_ng'][0])
    assert skinfo.joint_entropy(g, n, bins='fd') == pytest.approx(matlab_res['joint_entropy_ng'][0])

    return True






def test_skinfo_metrics():
    """
    In these tests, measures of information theory calculated with skinfo 
    are compared with the same measures calculated with Matlab.
    The test data is generated with numpy. Input data for the Matlab code 
    are mostly pdfs of this data with different binnings.
    """

    # generate data
    np.random.seed(42)
    n = np.random.normal(40, 2.5, 1000)
    np.random.seed(2409)
    g = np.random.gamma(10, 7, 1000)

    # load results from matlab code
    # all results are calculated with bin setting 'fd'
    matlab_res = pd.read_csv('skinfo/test/results_matlab.csv')
    matlab_res = matlab_res.to_dict(orient='list')

    # run single tests
    assert entropy_test(n, g, matlab_res)
    assert conditional_entropy_test(n, g, matlab_res)
    assert mutual_information_test(n, g, matlab_res)
    assert cross_entropy_test(n, g, matlab_res)
    assert joint_entropy_test(n, g, matlab_res)

