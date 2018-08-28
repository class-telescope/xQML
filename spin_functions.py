"""
Set of routines to compute the pixel covariance matrix using
Legendre polynomials

Author: Vanneste
"""
from __future__ import division

import math

import numpy as np
from scipy import special


def dlss(z, s1, s2, lmax):
    """
    Computes the reduced Wigner D-function d^l_ss'

    Parameters
    ----------
    z : float
        Cosine of the angle between two pixels
    s1 : int
        Spin number 1
    s2 : int
        Spin number 2
    lmax : int
        Maximum multipole

    Returns
    ----------
    d : 1D array of floats
        ???

    Example
    ----------
    >>> d = dlss(0.1, 2,  2, 5)
    >>> print(round(sum(d),5))
    0.24351
    """
    d = np.zeros((lmax + 1))
    if s1 < abs(s2):
        print("error spins, s1<|s2|")
        return

    # Conv: sign = -1 if (s1 + s2) and 1 else 1
    sign = (-1)**(s1 - s2)
    fs1 = math.factorial(2.0 * s1)
    fs1ps2 = math.factorial(1.0 * s1 + s2)
    fs1ms2 = math.factorial(1.0 * s1 - s2)
    num = (1.0 + z)**(0.5 * (s1 + s2)) * (1.0 - z)**(0.5 * (s1 - s2))

    # Initialise the recursion (l = s1 + 1)
    d[s1] = sign / 2.0**s1 * np.sqrt(fs1 / fs1ps2 / fs1ms2) * num

    l1 = s1 + 1.0
    rhoSSL1 = np.sqrt((l1 * l1 - s1 * s1) * (l1 * l1 - s2 * s2)) / l1
    d[s1+1] = (2 * s1 + 1.0)*(z - s2 / (s1 + 1.0)) * d[s1] / rhoSSL1

    # Build the recursion for l > s1 + 1
    for l in np.arange(s1 + 1, lmax, 1):
        l1 = l + 1.0
        numSSL = (l * l * 1.0 - s1 * s1) * (l * l * 1.0 - s2 * s2)
        rhoSSL = np.sqrt(numSSL) / (l * 1.0)
        numSSL1 = (l1 * l1 - s1 * s1) * (l1 * l1 - s2 * s2)
        rhoSSL1 = np.sqrt(numSSL1) / l1

        numd = (2.0 * l + 1.0) * (z - s1 * s2 / (l * 1.0) / l1) * d[l]
        d[l+1] = (numd - rhoSSL * d[l-1]) / rhoSSL1
    return d


def pl0(z, lmax):
    """
    Computes the associated Legendre function of the first kind of order 0
    Pn(z) from 0 to lmax (inclusive).

    Parameters
    ----------
    z : float
        Cosine of the angle between two pixels
    lmax : int
        Maximum multipole

    Returns
    ----------
    Pn : 1D array of floats
    Legendre function

    Example
    ----------
    >>> thepl0 = pl0(0.1, 5)
    >>> print(round(sum(thepl0),5))
    0.98427
    """
    Pn = special.lpn(lmax, z)[0]
    return Pn


def pl2(z, lmax):
    """
    Computes the associated Legendre function of the first kind of order 2
    from 0 to lmax (inclusive)

    Parameters
    ----------
    z : float
        Cosine of the angle between two pixels
    lmax : int
        Maximum multipole

    Returns
    ----------
    Pn2 : 1D array of floats
    Legendre function

    Example
    ----------
    >>> thepl2 = pl2(0.1, 5)
    >>> print(round(sum(thepl2),5))
    -7.49183
    """
    Pn2 = special.lpmn(2, lmax, z)[0][2]
    return Pn2


# ####### F1 and F2 functions from Tegmark & De Oliveira-Costa, 2000  #########
def F1l0(z, lmax):
    """
    Compute the F1l0 function from Tegmark & De Oliveira-Costa, 2000

    Parameters
    ----------
    z : float
        Cosine of the angle between two pixels
    lmax : int
        Maximum multipole

    Returns
    ----------
    bla : 1D array of float
    F1l0 function from Tegmark & De Oliveira-Costa, 2000

    Example
    ----------
    >>> theF1l0= F1l0(0.1, 5)
    >>> print(round(sum(theF1l0),5))
    0.20392
    """
    if abs(z) == 1.0:
        return(np.zeros(lmax + 1))
    else:
        ell = np.arange(2, lmax + 1)
        thepl = pl0(z, lmax)
        theplm1 = np.append(0, thepl[:-1])
        thepl = thepl[2:]
        theplm1 = theplm1[2:]
        a0 = 2.0 / np.sqrt((ell - 1) * ell * (ell + 1) * (ell + 2))
        a1 = ell * z * theplm1 / (1 - z**2)
        a2 = (ell / (1 - z**2) + ell * (ell - 1) / 2) * thepl
        bla = np.append([0, 0], a0 * (a1 - a2))
        return bla


def F1l2(z, lmax):
    """
    Compute the F1l2 function from Tegmark & De Oliveira-Costa, 2000

    Parameters
    ----------
    z : float
        Cosine of the angle between two pixels
    lmax : int
        Maximum multipole

    Returns
    ----------
    bla : 1D array of float
    F1l2 function from Tegmark & De Oliveira-Costa, 2000

    Example
    ----------
    >>> theF1l2= F1l2(0.1, 5)
    >>> print(round(sum(theF1l2),5))
    0.58396
    """
    if z == 1.0:
        return np.append(np.zeros(2), np.ones(lmax - 1) * 0.5)
    elif z == -1.0:
        ell = np.arange(lmax + 1)
        return np.append(np.zeros(2), 0.5 * (-1)**ell[2:])
    else:
        ell = np.arange(2, lmax + 1)
        thepl2 = pl2(z, lmax)
        theplm1_2 = np.append(0, thepl2[:-1])
        thepl2 = thepl2[2:]
        theplm1_2 = theplm1_2[2:]
        a0 = 2.0 / ((ell - 1) * ell * (ell + 1) * (ell + 2))
        a1 = (ell + 2) * z * theplm1_2 / (1 - z**2)
        a2 = ((ell - 4) / (1 - z**2) + ell * (ell - 1) / 2) * thepl2
        bla = np.append([0, 0], a0 * (a1 - a2))
        return bla


def F2l2(z, lmax):
    """
    Compute the F2l2 function from Tegmark & De Oliveira-Costa, 2000

    Parameters
    ----------
    z : float
        Cosine of the angle between two pixels
    lmax : int
        Maximum multipole

    Returns
    ----------
    ???

    Example
    ----------
    >>> theF2l2= F2l2(0.1, 5)
    >>> print(round(sum(theF2l2),5))
    0.34045
    """
    if z == 1.0:
        return np.append(np.zeros(2), -0.5 * np.ones(lmax - 1))
    elif z == -1.0:
        ell = np.arange(lmax + 1)
        return np.append(np.zeros(2), 0.5 * (-1)**ell[2:])
    else:
        ell = np.arange(2, lmax + 1)
        thepl2 = pl2(z, lmax)
        theplm1_2 = np.append(0, thepl2[:-1])
        thepl2 = thepl2[2:]
        theplm1_2 = theplm1_2[2:]
        a0 = 4.0 / ((ell - 1) * ell * (ell + 1) * (ell + 2) * (1 - z**2))
        a1 = (ell + 2) * theplm1_2
        a2 = (ell - 1) * z * thepl2
        bla = np.append([0, 0], a0 * (a1 - a2))
        return bla

if __name__ == "__main__":
    """
    Run the doctest using

    python simulation.py

    If the tests are OK, the script should exit gracefuly, otherwise the
    failure(s) will be printed out.
    """
    import doctest
    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")
    doctest.testmod()
