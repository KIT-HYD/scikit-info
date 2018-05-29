import numpy as np


def entropy(x, bins=10):
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
        In all other cases, bins will be passed through to numpy.histogram in
        order to calculate the bin edges

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
    # calculate the empirical probabilities
    count = np.histogram(x, bins=bins)[0]
    p = (count / np.sum(count)) + 1e-15

    # calculate the Shannon Entropy
    return - p.dot(np.log2(p))


def conditional_entropy(x, y, bins):
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
        In all other cases, bins will be passed through to numpy.histogram in
        order to calculate the bin edges

    Returns
    -------
    float

    Notes
    -----

    This implementation will only work if x and y are of same length.

    """
    assert len(x) == len(y)

    # calculate the joint distribution and marginal distribution for y
    joint_count = np.histogram2d(x, y, bins=bins)[0]
    marginal_y = np.histogram(y, bins=bins)[0]

    # generate an iterator for each bin in x
    def marg(joint, marginal):
        for i in range(len(marginal)):
            # add 1e-15 to supress log2(0)
            p_i = joint[:, i] / np.sum(joint[:, i] + 1e-15) + 1e-15
            H_i = - p_i.dot(np.log2(p_i))

            # get the probability for the whole bin
            pxi = marginal[i] / np.sum(marginal) # zeros are ok

            yield pxi * H_i

    # conditional entropy is the sum of all Entropies for the single bins
    hxy = np.fromiter((m for m in marg(joint_count, marginal_y)), dtype=float)
    return np.sum(hxy)


def mutual_information(x, y, bins):
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
        In all other cases, bins will be passed through to numpy.histogram in
        order to calculate the bin edges

    Returns
    -------
    float

    Notes
    -----
    The mutual information is defined to be  the difference of the entropy of X
    and the conditional entropy of X given Y.

    .. math::

        I(X;Y) = H(X) - H(X|Y)

    This implementation will only work if x and y are of same length.

    """
    hx = entropy(x, bins=bins)
    hcon = conditional_entropy(x, y, bins=bins)

    return hx - hcon


