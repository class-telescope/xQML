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
    """
    pixarea = hp.nside2pixarea(nside, degrees = True)
    varperpix = (muKarcmin*1e-6/60.)**2/pixarea
    return varperpix

def pixvar2nl(pixvar, nside):
    """
    Return noise spectrum level for a given nside and  pixel variance

    Parameters
    ----------
    pixvar : ???
        ???
    ...

    Returns
    ----------
    ??? : ???
        ???
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
