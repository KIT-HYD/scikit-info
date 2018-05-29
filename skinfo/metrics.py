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
        The speicfication for the bin edges used to calculate the Entropy.
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
