"""
Set of routines to ...

Author: Vanneste
"""
from __future__ import division

import math

import numpy as np
from scipy import special

def dlss(z, s1, s2, lmax):
    """
    Matt version ???

    Parameters
    ----------
    z : ???
        ???
    s1 : int
        Spin number ???
    s2 : int
        Spin number ???
    lmax : int
        Maximum multipole

    Returns
    ----------
    d : 1D array of floats
        ???

    """
    d = np.zeros((lmax + 1))
    if s1 < abs(s2):
        print("error spins, s1<|s2|")
        return

    # sign = -1 if (s1 + s2) and 1 else 1
    sign = (-1)**(s1 - s2)
    fs1 = math.factorial(2.0 * s1)
    fs1ps2 = math.factorial(1.0 * s1 + s2)
    fs1ms2 = math.factorial(1.0 * s1 - s2)
    num = (1.0 + z)**(0.5 * (s1 + s2)) *  (1.0 - z)**(0.5 * (s1 - s2))

    # Initialise the recursion (l = s1 + 1)
    d[s1] = sign / 2.0**s1 * np.sqrt(fs1 / fs1ps2 / fs1ms2) * num

    l1 = s1 + 1.0
    rhoSSL1 = np.sqrt((l1 * l1 - s1 * s1) * (l1 * l1 - s2 * s2)) / l1
    d[s1+1] = (2 * s1 + 1.0)*(z - s2 / (s1 + 1.0)) * d[s1] / rhoSSL1

    # Build the recursion for l > s1 + 1
    for l in np.arange(s1 + 1, lmax, 1) :
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
    Compute sequence of Legendre functions of the first kind (polynomials),
    Pn(z) and derivatives for all degrees from 0 to lmax (inclusive).

    Parameters
    ----------
    z : ???
        ???
    lmax : int
        Maximum multipole

    Returns
    ----------
    ???

    """
    # 0 is for no derivative?
    return special.lpn(lmax, z)[0]

def pl2(z, lmax):
    """
    Computes the associated Legendre function of the first kind of order m and
    degree n, ``Pmn(z)`` = :math:`P_n^m(z)`, and its derivative

    Parameters
    ----------
    z : ???
        ???
    lmax : int
        Maximum multipole

    Returns
    ----------
    ???
    """
    return(special.lpmn(2, lmax, z)[0][2])


######## F1 and F2 functions from Tegmark & De Oliveira-Costa, 2000  ##########
def F1l0(z, lmax):
    """
    ???

    Parameters
    ----------
    z : ???
        ???
    lmax : int
        Maximum multipole

    Returns
    ----------
    ???
    """
    if abs(z) == 1.0:
        return(np.zeros(lmax + 1))
    else:
        ell = np.arange(lmax + 1)
        thepl = pl0(z, lmax)
        theplm1 = np.append(0, pl0(z, lmax - 1))
        a0 = 2.0 / np.sqrt((ell - 1) * ell * (ell + 1) * (ell + 2))
        a1 = ell * z * theplm1 / (1 - z**2)
        a2 = (ell / (1 - z**2) + ell * (ell - 1) / 2) * thepl
        bla = a0 * (a1 - a2)
        # deux premiers poles
        bla[0] = 0.0
        bla[1] = 0.0

        return bla

def F1l2(z, lmax):
    """
    ???

    Parameters
    ----------
    z : ???
        ???
    lmax : int
        Maximum multipole

    Returns
    ----------
    ???
    """
    if z == 1.0:
        # former version
        # return(np.ones(lmax+1)*0.5)
        # = 0 pour l=0,1
        return np.append(np.zeros(2), np.ones(lmax - 1) * 0.5)
    elif z == -1.0:
        ell = np.arange(lmax + 1)
        # former version
        # return(0.5*(-1)**ell)
        # = 0 pour l=0,1
        return np.append(np.zeros(2), 0.5 * (-1)**ell[2:])
    else:
        ell = np.arange(lmax + 1)
        thepl2 = pl2(z, lmax)
        theplm1_2 = np.append(0, pl2(z, lmax - 1))
        a0 = 2.0 / ((ell - 1) * ell * (ell + 1) * (ell + 2))
        a1 = (ell + 2) * z * theplm1_2 / (1 - z**2)
        a2 = ((ell - 4) / (1 - z**2) + ell * (ell - 1) / 2) * thepl2
        bla = a0 * (a1 - a2)
        # deux premiers poles
        bla[0] = 0.0
        bla[1] = 0.0

        return bla

def F2l2(z,lmax):
    """
    ???

    Parameters
    ----------
    z : ???
        ???
    lmax : int
        Maximum multipole

    Returns
    ----------
    ???
    """
    if z == 1.0:
        # former version
        # return(-0.5*np.ones(lmax+1))
        # = 0 pour l=0,1
        return np.append(np.zeros(2), -0.5 * np.ones(lmax - 1))
    elif z == -1.0:
        ell = np.arange(lmax + 1)
        # former version
        # return(0.5*(-1)**ell)
        # syl : = 0 pour l=0,1
        return np.append(np.zeros(2), 0.5 * (-1)**ell[2:])
    else:
        ell = np.arange(lmax + 1)
        thepl2 = pl2(z, lmax)
        theplm1_2 = np.append(0, pl2(z, lmax - 1))
        a0 = 4.0 / ((ell - 1) * ell * (ell + 1) * (ell + 2) * (1 - z**2))
        a1 = (ell + 2) * theplm1_2
        a2 = (ell - 1) * z * thepl2
        bla = a0 * (a1 - a2)
        # ??
        bla[0] = 0
        bla[1] = 0

        return bla
