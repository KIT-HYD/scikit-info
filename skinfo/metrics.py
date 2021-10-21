import numpy as np


def entropy(x, bins, normalize=False, use_probs=False):
    """ Shannon Entropy
    Calculates the Shannon Entropy for the given data array x.

    Parameters
    ----------
    x : numpy.ndarray
        Array of observations that should be used to calculate the Entropy of
        their empirical distribution density function.
    bins : integer, list, array, string
        The specification for the bin edges used to calculate the Entropy.
        In case bins is a list, the list members will be used as bin edges.
        In all other cases, bins will be passed through to 
        numpy.histogram_bin_edges in order to calculate the bin edges.
    normalize: bool
        If normalize is True, the entropy is normalized by the maximum entropy.
        Defaults to False.
    use_probs: bool
        If True, it is assumed that x is already an empirical probability 
        distribution, not observations.
        Defaults to False.

    Returns
    -------
    float

    Notes
    -----
    The formula used for Shannon Entropy H(x) is:

    .. math::

        H(x) = - \sum_{i=1}^N p(x_i) * log_2(p(x_i))

    where N is the number of bins, and x_i all observations falling into the
    bin i.

    In  order to prevent the logarithm to evaluate to infinity, 1e-15 is
    added to all probabilities. These should even out, as the information
    content is multiplied with this small number. However, the sum of all
    probailities calculated this way might not be exactly 1.

    """
    # calculate probabilities if use_probs == False
    if use_probs:
        # if x does not sum up to 1, raise an error
        if not np.isclose(sum(x),1,atol=0.0001):
            raise ValueError('Probabilities in vector x do not sum up to 1.')
        
        p = x + 1e-15
    else:
        # get the bins
        bins = np.histogram_bin_edges(x, bins)

        # calculate the empirical probabilities
        count = np.histogram(x, bins=bins)[0]

        # if counts should be None, raise an error
        if np.sum(count) == 0:
            raise ValueError('The histogram cannot be empty. Adjust the bins to ' +
                             'fit the data')
        # calculate the probabilities
        p = (count / np.sum(count)) + 1e-15


    # calculate the Shannon Entropy
    if normalize:
        # get number of bins
        nbins = len(p)
        # maximal entropy: uniform distribution
        normalizer = np.log2(len(nbins)) 

        return - p.dot(np.log2(p)) / normalizer
    else:
        return - p.dot(np.log2(p))


def conditional_entropy(x, y, bins, normalize=False):
    """ Conditional Entropy

    Calculates the conditional Shannon Entropy for two discrete
    distributions. This metric gives the entropy of the distribution of x in
    case the distribution of y is known.


    Parameters
    ----------
    x : numpy.ndarray
        Array of observations that should be used to calculate the
        conditional entropy of their empirical distribution density function
        given y.
    y : numpy.ndarray
        Array of observations that are used as the known distribution to
        calculate the entropy of x conditioned to y.
    bins : integer, list, array, string
        The specification for the bin edges used to calculate the Entropy.
        In case bins is a list, the list members will be used as bin edges.
        In all other cases, bins will be passed through to 
        numpy.histogram_bin_edges in order to calculate the bin edges.
    normalize: bool
        If normalize is True, the conditional entropy is normalized by division
        by the entropy of x, as the conditional entropy of x|y is never
        greater than the entropy of x alone.
        Defaults to False.

    Returns
    -------
    float

    Notes
    -----

    The conditional entropy is calculated using the joint entropy as follows:

    .. math::

        H(x|y) = H(x,y) - H(y)

    where H(x,y) is the joint entropy of x and y; H(y) is the unconditioned
    entropy of y.

    """
    # calculate H(x,y) and H(y)
    hjoint = joint_entropy(x,y,bins)
    hy = entropy(y, bins)

    if normalize:
        bins_x = np.histogram_bin_edges(x, bins)
        normalizer = entropy(x, bins_x)
        return (hjoint - hy) / normalizer
    else:
        return hjoint - hy


def mutual_information(x, y, bins, normalize=False):
    """ Mutual information

    Calculates the mutual information of a discrete distribution x given a
    known discrete distribution y. The mutual information is the amount of
    information that two distributions share.

    Parameters
    ----------
    x : numpy.ndarray
        Array of observations that should be used to calculate the
        mututal information shared by two discrete distributions x and y.
    y : numpy.ndarray
        See x.
    bins : integer, list, array, string
        The specification for the bin edges used to calculate the Entropy.
        In case bins is a list, the list members will be used as bin edges.
        In all other cases, bins will be passed through to 
        numpy.histogram_bin_edges in order to calculate the bin edges
    normalize: bool
        If normalize is True, the mutual information is normalized by division
        by the entropy of x or y, depending on which is smaller, as the mutual 
        information of x and y is never greater than the entropy of x or y.
        Defaults to False.

    Returns
    -------
    float

    Notes
    -----
    The mutual information is defined to be the difference of the entropy of X
    and the conditional entropy of X given Y.

    .. math::

        I(X;Y) = H(X) - H(X|Y)

    This implementation will only work if x and y are of same length.

    """
    # assert array length
    assert len(x) == len(y)

    # calculate entropy(x) and conditional_entropy(x,y)
    hx = entropy(x, bins)
    hcon = conditional_entropy(x, y, bins)

    if normalize:
        # get the bins
        bins_x = np.histogram_bin_edges(x, bins)
        bins_y = np.histogram_bin_edges(y, bins)

        normalizer = np.min([entropy(x, bins_x), entropy(y, bins_y)])
        return (hx - hcon) / normalizer
    else:
        return hx - hcon

      
def cross_entropy(x, y, bins, use_probs=False):
    """ Cross Entropy

    Calculates the cross entropy of two discrete distributions x and y.

    Parameters
    ----------
    x : numpy.ndarray
        Array of observations that should be used to calculate the
        cross entropy of two discrete distributions x and y.
    y : numpy.ndarray
        See x.
    bins : integer, list, array, string
        The specification for the bin edges used to calculate the Entropy.
        In case bins is a list, the list members will be used as bin edges.
        In all other cases, bins will be passed through to 
        numpy.histogram_bin_edges in order to calculate the bin edges
    use_probs: bool
        If True, it is assumed that x and y are already a empirical probability 
        distributions, not observations.
        Note, that the empirical probability distributions for x and y have to 
        be calculated with the same bins for cross entropy.
        Defaults to False.

    Returns
    -------
    float

    Notes
    -----

    The cross entropy is defined as the sum of all information contents
    inherit to the bins in y multiplied by the probability of the same bin in x:

    .. math::

        H(x||y) = - \sum_x p(x) * log_2 [p(y)]

    """
   # calculate probabilities if probabilities == False
    if use_probs:
        # same bins for x and y -> same length of x and y if use_probs == True
        assert len(x) == len(y)

        # if x does not sum up to 1, raise an error
        if not np.isclose(sum(x),1,atol=0.0001):
            raise ValueError('Probabilities in vector x do not sum up to 1.')
        # if y does not sum up to 1, raise an error
        if not np.isclose(sum(y),1,atol=0.0001):
            raise ValueError('Probabilities in vector y do not sum up to 1.')

        px = x + 1e-15
        py = y + 1e-15
    else:
        # get the bins
        bins = np.histogram_bin_edges([x, y], bins)

        # calculate unconditioned histograms
        hist_x = np.histogram(x, bins=bins)[0]
        hist_y = np.histogram(y, bins=bins)[0]

        px = (hist_x / np.sum(hist_x)) + 1e-15
        py = (hist_y / np.sum(hist_y)) + 1e-15

    return - px.dot(np.log2(py))


def joint_entropy(x, y, bins):
    r"""Joint Entropy

    Calculates the joint entropy of two discrete distributions x and y. This
    is the combined Entropy of X added to the conditional Entropy of x giv y.
    The joint entropy will use the same bin setting for both distributions.

    Parameters
    ----------
    x : numpy.ndarray
        Array of observations that should be used to calculate the
        joint entropy between two discrete distributions x and y.
    y : numpy.ndarray
        See x.
    bins : integer, list, array, string
        The specification for the bin edges used to calculate the Entropy.
        In case bins is a list, the list members will be used as bin edges.
        In all other cases, bins will be passed through to 
        numpy.histogram_bin_edges in order to calculate the bin edges

    Returns
    -------
    float

    Notes
    -----

    The joint entropy is defined to be the sum of the entropy of y and the
    conditional entropy of x given y.

    .. math::

        H(x,y) = -\sum_x \sum_y P(x,y) * log_2[P(x,y)]

    """
    # assert array length
    assert len(x) == len(y)

    # get the bins, x and y get their own bins in case of joint entropy
    bins_x = np.histogram_bin_edges(x, bins)
    bins_y = np.histogram_bin_edges(y, bins) 
    bins_xy = [bins_x, bins_y]

    # get the joint histogram
    joint_hist = np.histogram2d(x, y, bins_xy)[0]

    # calculate the joint probability and add a small number
    joint_p = (joint_hist / np.sum(joint_hist)) + 1e-15

    # calculate and return the joint entropy
    return - np.sum(joint_p * np.log2(joint_p))


def kullback_leibler(x, y, bins, use_probs=False):
    r"""Kullback-Leibler Divergence

    Calculates the Kullback-Leibler Divergence between two discrete
    distributions x and y. X is considered to be an empirical discrete
    distribution while y is considered to be the real discrete distribution
    of the underlying population.

    Parameters
    ----------
    x : numpy.ndarray
        Array of observations that should be used to calculate the
        Kullback-Leibler divergence between two discrete distributions x and y.
    y : numpy.ndarray
        See x.
    bins : integer, list, array, string
        The specification for the bin edges used to calculate the Entropy.
        In case bins is a list, the list members will be used as bin edges.
        In all other cases, bins will be passed through to 
        numpy.histogram_bin_edges in order to calculate the bin edges
    use_probs: bool
        If True, it is assumed that x and y are already empirical probability 
        distributions, not observations.
        Note, that the empirical probability distributions for x and y have to 
        be calculated with the same bins for Kullback-Leibler divergence.
        Defaults to False.

    Returns
    -------
    float

    Notes
    -----

    The Kullback-Leibler divergence is calculated as the difference of the
    cross entropy between x and y and the unconditioned entropy of x:

    .. math::

        D_{KL}(x||y) = H(x||y) - H(x)

    """
    if use_probs:
        # if x does not sum up to 1, raise an error
        if not np.isclose(sum(x),1,atol=0.0001):
            raise ValueError('Probabilities in vector x do not sum up to 1.')
        # if y does not sum up to 1, raise an error
        if not np.isclose(sum(y),1,atol=0.0001):
            raise ValueError('Probabilities in vector y do not sum up to 1.')

    # calculte the cross entropy and unconditioned entropy of y
    hcross = cross_entropy(x, y, bins, use_probs=use_probs)
    hx = entropy(x, bins, use_probs=use_probs)
    
    return hcross - hx

def jensen_shannon(x, y, bins, calc_distance=False, use_probs=False):
    r"""Jensen-Shannon Divergence

    Calculates the Jensen-Shannon Divergence (JSD) between two discrete
    distributions x and y. JSD quantifies the difference (or similarity) 
    between two probability distributions and uses the KL divergence to 
    calculate a smoothed normalized score [0, 1] that is symmetrical.

    Parameters
    ----------
    x : numpy.ndarray
        Array of observations that should be used to calculate the
        Jensen-Shannon divergence between two discrete distributions x and y.
    y : numpy.ndarray
        See x.
    bins : integer, list, array, string
        The specification for the bin edges used to calculate the Entropy.
        In case bins is a list, the list members will be used as bin edges.
        In all other cases, bins will be passed through to 
        numpy.histogram_bin_edges in order to calculate the bin edges
    calc_distance : bool
        If True, the Jensen-Shannon distance instead of the Jensen-Shannon
        divergence is returned.
        Jensen-Shannon distance is a metric and is the square root of the 
        Jensen-Shannon divergence.
        Defaults to False.
    use_probs: bool
        If True, it is assumed that x and y are already empirical probability 
        distributions, not observations.
        Note, that the empirical probability distributions for x and y have to 
        be calculated with the same bins for Jensen-Shannon divergence / distance.
        Defaults to False.
    Returns
    -------
    float

    Notes
    -----

    The Jensen-Shannon Divergence is based on the Kullback-Leibler Divergence
    and is calculated as follows:

    .. math::

        JSD(x||y) = 1/2 * D_{KL}(p_{x}||p_{m}) + 1/2 * D_{KL}(p_{y}||p_{m})

        p_{m} = 1/2 * (p_{x} + p_{y})

    References
    -----
    B. Fuglede and F. Topsoe, "Jensen-Shannon divergence and Hilbert 
    space embedding," International Symposium onInformation Theory, 2004. 
    ISIT 2004. Proceedings., 2004, pp. 31-, doi: 10.1109/ISIT.2004.1365067.
    """
    # assert array length
    assert len(x) == len(y)

    if use_probs:
        # if x does not sum up to 1, raise an error
        if not np.isclose(sum(x), 1 ,atol=0.0001):
            raise ValueError('Probabilities in vector x do not sum up to 1.')
        # if y does not sum up to 1, raise an error
        if not np.isclose(sum(y), 1, atol=0.0001):
            raise ValueError('Probabilities in vector y do not sum up to 1.')

        px = x + 1e-15
        py = y + 1e-15
    else:
        # get the bins
        bins = np.histogram_bin_edges([x, y], bins)

        # calculate unconditioned histograms
        hist_x = np.histogram(x, bins=bins)[0]
        hist_y = np.histogram(y, bins=bins)[0]

        # calculate probabilities
        px = (hist_x / np.sum(hist_x)) + 1e-15
        py = (hist_y / np.sum(hist_y)) + 1e-15

    # calculate m
    pm = 0.5 * (px + py)

    # calculate kullback-leibler divergence between px and pm & py and pm
    kl_xm = kullback_leibler(px, pm, bins=bins, use_probs=True)
    kl_ym = kullback_leibler(py, pm, bins=bins, use_probs=True)
    
    if calc_distance:
        return (0.5 * kl_xm + 0.5 * kl_ym)**0.5
    else:
        return (0.5 * kl_xm + 0.5 * kl_ym)