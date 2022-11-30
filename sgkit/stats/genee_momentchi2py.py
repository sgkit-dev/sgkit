# Copyright (c) 2018 momentchi2py authors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
from scipy.stats import gamma


# This function duplicates momentchi2py's hbe function from
# https://github.com/deanbodenham/momentchi2py.
#
# Hall-Buckley-Eagleson
def hbe(coeff, x):  # pragma: no cover
    """Hall-Buckley-Eagleson method
    Computes the cdf of a positively-weighted sum of chi-squared
    random variables with the Hall-Buckley-Eagleson (HBE) method.
    Parameters:
    -----------
    coeff (list or numpy array): The coefficient vector.
                                 All values must be greater than 0.
    x (list or numpy array or float): The vector of quantile values.
                                      All values must be greater than 0.
    Returns:
    --------
    The cdf of the x value(s). It is returned as the same type as x,
    i.e. if x is a list, it is returned as a list; if x is a numpy array
    it is returned as a numpy array.
    Details:
    --------
     * Note that division assumes Python 3, so may not work with Python 2.
     * Depends on numpy libary for the arrays.
     * If lists are passed, they are converted to numpy arrays (and back again).
     * Depends on scipy library for scipy.stats.gamma function.
    Examples:
    ---------
    #Examples taken from Table 18.6 in N. L. Johnson, S. Kotz, N. Balakrishnan.
    #Continuous Univariate Distributions, Volume 1, John Wiley & Sons, 1994.
    # how to load the hbe function from momenthchi2
    from momentchi2 import hbe
    # should give value close to 0.95, actually 0.94908
    hbe([1.5, 1.5, 0.5, 0.5], 10.203)
    # should give value close to 0.05, but is actually 0.02857
    hbe([1.5, 1.5, 0.5, 0.5], 0.627)
    # specifying parameters
    hbe(coeff=[1.5, 1.5, 0.5, 0.5], x=10.203)
    # x is a list, output approx. 0.05, 0.95
    hbe([1.5, 1.5, 0.5, 0.5], [0.627, 10.203])
    # instead of lists can be numpy arrays
    # (any list is converted to a numpy arrays inside the function anyway)
    import numpy as
    hbe( np.array([1.5, 1.5, 0.5, 0.5]), np.array([0.627, 10.203]) )
    References:
    -----------
     * P. Hall. Chi squared approximations to the distribution of a
       sum of independent random variables. The Annals of
       Probability, 11(4):1028-1036, 1983.
     * M. J. Buckley and G. K. Eagleson. An approximation to the
       distribution of quadratic forms in normal random variables.
       Australian Journal of Statistics, 30(1):150-159, 1988.
     * D. A. Bodenham and N. M. Adams. A comparison of efficient
       approximations for a weighted sum of chi-squared random variables.
       Statistics and Computing, 26(4):917-928, 2016.
    """
    # some checking, so that passing lists/arrays does not matter
    if isinstance(coeff, list):
        coeff = np.array(coeff)

    isList = False
    if not isinstance(x, float):
        if isinstance(x, list):
            isList = True
            x = np.array(x)

    # checking values of coeff and x and throwing errors
    if not checkCoeffAllPositive(coeff):
        raise Exception(getCoeffError(coeff))

    if not checkXAllPositive(x):
        raise Exception(getXError(x))

    # the next two lines work for floats or numpy arrays, but not lists
    K_1 = sum(coeff)
    K_2 = 2 * sum(coeff**2)
    K_3 = 8 * sum(coeff**3)
    nu = 8 * (K_2**3) / (K_3**2)
    k = nu / 2
    theta = 2
    # in the next line x can be a float or numpy array, but not a list
    x = ((2 * nu / K_2) ** (0.5)) * (x - K_1) + nu
    p = gamma.cdf(x, a=k, scale=theta)

    # if x was passed as a list, will return a list
    if isList:
        p = p.tolist()
    return p


def checkCoeffAllPositive(coeff):  # pragma: no cover
    """check all entries in coeff vector positive"""
    if isinstance(coeff, (int, float)):
        return coeff > 0
    return all(i > 0 for i in coeff)


def getCoeffError(coeff):  # pragma: no cover
    """get the error message if there is a coeff error"""
    if isinstance(coeff, (int, float)):
        if not (coeff > 0):
            return "coefficient value needs to be strictly > 0."
    else:
        if not (all(i > 0 for i in coeff)):
            return "all coefficients need to be strictly > 0."
    return "unknown error with coefficients."


def checkXAllPositive(x):  # pragma: no cover
    """check all entries in x vector positive"""
    if isinstance(x, (int, float)):
        return x > 0
    return all(i > 0 for i in x)


def getXError(x):  # pragma: no cover
    """get the error message if there is an error in x vector"""
    if isinstance(x, (int, float)):
        if not (x > 0):
            return "x value needs to be strictly > 0."
    else:
        if not (all(i > 0 for i in x)):
            return "all values in x need to be strictly > 0."
    return "unknown error with x."
