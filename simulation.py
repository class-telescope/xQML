"""
Set of routines to ...

Author: Vanneste
"""
from __future__ import division

import numpy as np
import healpy as hp

from scipy import sparse

def muKarcmin2var(muKarcmin, nside):
    """
    Return pixel variance for a given nside and noise level [1e-6 K . arcmin]

    Parameters
    ----------
    muKarcmin : ???
        ???
    ...

    Returns
    ----------
    ??? : ???
        ???


    Example
    ----------
    >>> var = muKarcmin2var(10.0, 16)
    >>> print(round(var * 1e16))
    21.0
    """
    pixarea = hp.nside2pixarea(nside, degrees=True)
    varperpix = (muKarcmin * 1e-6 / 60.)**2 / pixarea
    return varperpix

def pixvar2nl(pixvar, nside):
    """
    Return noise spectrum level for a given nside and pixel variance

    Parameters
    ----------
    pixvar : ???
        ???
    ...

    Returns
    ----------
    ??? : ???
        ???

    Example
    ----------
    >>> nside = 16
    >>> pixvar = muKarcmin2var(10.0, nside)
    >>> nl = pixvar2nl(pixvar, nside)
    >>> print(round(nl * 1e18))
    8.0
    """
    return pixvar*4.*np.pi/(12*nside**2.)

def getNl(pixvar, nside, nbins):
    """
    Return noise spectrum for a given nside and pixel variance

    Parameters
    ----------
    pixvar : ???
        ???
    ...

    Returns
    ----------
    ??? : ???
        ???

    Example
    ----------
    >>> nside = 16
    >>> pixvar = muKarcmin2var(10.0, nside)
    >>> nl = getNl(pixvar, nside, 2)
    >>> print(round(nl[0] * 1e18))
    8.0
    """
    return pixvar*4.*np.pi/(12*nside**2.)*np.ones((nbins))


def getstokes(polar=True, temp=False, EBTB=False):
    """
    ???

    Parameters
    ----------
    polar : ???
        ???
    ...

    Returns
    ----------
    ??? : ???
        ???

    Example
    ----------
    >>> getstokes(polar=True, temp=False, EBTB=False)
    (['Q', 'U'], ['EE', 'BB'], [1, 2])
    >>> getstokes(polar=True, temp=True, EBTB=False)
    (['I', 'Q', 'U'], ['TT', 'EE', 'BB', 'TE'], [0, 1, 2, 3])
    >>> getstokes(polar=True, temp=True, EBTB=True)
    (['I', 'Q', 'U'], ['TT', 'EE', 'BB', 'TE', 'EB', 'TB'], [0, 1, 2, 3, 4, 5])
    """
    allStoke = ['I', 'Q', 'U']
    if EBTB:
        der = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
        ind = [0, 1, 2, 3, 4, 5]
    else:
        der = ['TT', 'EE', 'BB', 'TE']
        ind = [0, 1, 2, 3]
    if not temp:
        allStoke = ['Q', 'U']
        if EBTB:
            der = ['EE', 'BB', 'EB']
            ind = [1, 2, 4]
        else:
            der = ['EE', 'BB']
            ind = [1, 2]
    if not polar:
        allStoke = ['I']
        der = ['TT']
        ind = [0]
    return allStoke, der, ind

def GetBinningMatrix(
        ellbins, lmax, norm=False, polar=True,
        temp=False, EBTB=False, verbose=False):
    """
    Return P and Q matrices such taht Cb = P.Cl and Vbb = P.Vll.Q
    Return ell (total non-binned multipole range)
    Return ellval (binned multipole range)

    Parameters
    ----------
    ellbins : ???
        ???
    ...

    Returns
    ----------
    ??? : ???
        ???

    Example
    ----------
    >>> bins = np.array([2.0, 5.0, 10.0])
    >>> P, Q, ell, ellval = GetBinningMatrix(
    ...     bins, 10.0)
    >>> print(P) # doctest: +NORMALIZE_WHITESPACE
    [[ 0.33333333  0.33333333  0.33333333  0.          0.          0.       0.
       0.          0.          0.          0.          0.          0.       0.
       0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.2         0.2         0.2      0.2
       0.2         0.          0.          0.          0.          0.       0.
       0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.       0.
       0.          0.          0.33333333  0.33333333  0.33333333  0.       0.
       0.          0.          0.          0.        ]
     [ 0.          0.          0.          0.          0.          0.       0.
       0.          0.          0.          0.          0.          0.2      0.2
       0.2         0.2         0.2         0.        ]]
    >>> print(Q) # doctest: +NORMALIZE_WHITESPACE
    [[ 1.  0.  0.  0.]
     [ 1.  0.  0.  0.]
     [ 1.  0.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  1.  0.  0.]
     [ 0.  0.  0.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  1.  0.]
     [ 0.  0.  0.  1.]
     [ 0.  0.  0.  1.]
     [ 0.  0.  0.  1.]
     [ 0.  0.  0.  1.]
     [ 0.  0.  0.  1.]
     [ 0.  0.  0.  0.]]
    >>> print(ell)
    [  2.   3.   4.   5.   6.   7.   8.   9.  10.  11.]
    >>> print(ellval)
    [ 3.  7.]
    """
    #### define Stokes
    allStoke, der, ind = getstokes(polar, temp, EBTB)
    nder = len(der)

    nbins = len(ellbins) - 1
    ellmin = np.array(ellbins[0: nbins])
    ellmax = np.array(ellbins[1: nbins + 1]) - 1
    ell = np.arange(np.min(ellbins), lmax + 2)
    maskl = (ell[:-1] < (lmax + 2)) & (ell[:-1] > 1)

    # define min
    minell = np.array(ellbins[0: nbins])
    # and max of a bin
    maxell = np.array(ellbins[1: nbins + 1]) - 1
    ellval = (minell + maxell) * 0.5

    masklm = []
    for i in np.arange(nbins):
        masklm.append(((ell[:-1] >= minell[i]) & (ell[:-1] <= maxell[i])))

    allmasklm = nder*[list(masklm)]
    masklM = np.array(sparse.block_diag(allmasklm[:]).toarray())
    binsnorm = np.array(
        nder * [list(np.arange(minell[0], np.max(ellbins)))]).flatten()

    binsnorm = binsnorm*(binsnorm+1)/2./np.pi
    P = np.array(masklM)*1.
    Q = P.T
    P = P / np.sum(P, 1)[:, None]
    if norm:
        P *= binsnorm

    return P, Q, ell, ellval

def GetCorr(F):
    """
    ???

    Parameters
    ----------
    F : ???
        ???
    ...

    Returns
    ----------
    ??? : ???
        ???
    """
    nbins = len(F)
    Corr = np.array(
        [F[i, j] / (F[i, i]*F[j, j])**.5 for i in np.arange(nbins)
            for j in np.arange(nbins)]).reshape(nbins, nbins)
    return Corr

def IsInvertible(F):
    """
    ???

    Parameters
    ----------
    F : ???
        ???
    ...

    Returns
    ----------
    ??? : ???
        ???
    """
    eps = np.finfo(F.dtype).eps
    print("Cond Numb = ", np.linalg.cond(F), "Matrix eps=", eps)
    return np.linalg.cond(F) > np.finfo(F.dtype).eps


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
