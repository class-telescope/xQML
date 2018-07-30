"""
Set of routines to ...

Comment: what is Expd()? It is defined nowhere in the code...

Author: Vanneste
"""
from __future__ import division

import numpy as np

def Pl(ds_dcb):
    """
    Reshape ds_dcbin Pl

    Parameters
    ----------
    ds_dcb : ???
        ???

    Returns
    ----------
    ???
    """
    nnpix = np.shape(ds_dcb)[-1]
    return np.copy(ds_dcb).reshape(2 * (np.shape(ds_dcb)[1]), nnpix, nnpix)

def CorrelationMatrix(Clth, Pl, ellbins, polar=True, temp=False, EBTB=False):
    """
    Compute correlation matrix S = sum_l Pl*Cl

    Parameters
    ----------
    Clth : ???
        ???

    Returns
    ----------
    ???
    """
    if EBTB:
        xx = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
        ind = [0, 1, 2, 3, 4, 5]
    else:
        xx = ['TT', 'EE', 'BB', 'TE']
        ind = [0, 1, 2, 3]

    if not temp:
        allStoke = ['Q', 'U']
        if EBTB:
            xx = ['EE', 'BB', 'EB']
            ind = [1, 2, 5]
        else:
            xx = ['EE', 'BB']
            ind = [1, 2]
    if not polar:
        allStoke = ['I']
        xx = ['TT']
        ind = [0]

    clth = Clth[ind][:, 2: int(ellbins[-1])].flatten()
    return np.sum(Pl * clth[:, None, None], 0)

def El(invCAA, invCBB, Pl, Bll=None, expend=False):
    """
    Compute El = invCAA.Pl.invCBB

    Parameters
    ----------
    invCAA : ???
        ???

    Returns
    ----------
    ???
    """
    # Tegmark B-matrix useless so far)
    if Bll == None:
        Bll = np.diagflat((np.ones(len(Pl))))

    lmax = len(Pl) * 2**int(expend)
    lrange = np.arange(lmax)
    npix = len(invCAA)
    # triangular shape ds_dcb
    if expend:
        El = np.array([np.dot(np.dot(invCAA, Expd(Pl,l)), invCBB) for l in lrange]).reshape((lmax,npix,npix))
    else:
        El = np.array([np.dot(np.dot(invCAA, Pl[l]), invCBB) for l in lrange]).reshape((lmax,npix,npix))
    return El

def ElLong(invCAA, invCBB, Pl, Bll=None, expend=False):
    """
    Compute El = invCAA.Pl.invCBB

    Parameters
    ----------
    invCAA : ???
        ???

    Returns
    ----------
    ???
    """
    # Tegmark B-matrix useless so far)
    if Bll == None:
        Bll = np.diagflat((np.ones(len(Pl))))

    lmax = len(Pl) * 2**int(expend)
    lrange = np.arange(lmax)
    npix = len(invCAA)
    # triangular shape ds_dcb
    if expend:
        for l in lrange:
            Pl[l] = np.dot(np.dot(invCAA, Expd(Pl, l)), invCBB).reshape((npix, npix ))
    else:
        for l in lrange:
            Pl[l] = np.dot(np.dot(invCAA, Pl[l]), invCBB).reshape((npix, npix ))

def CrossFisherMatrix(El, Pl, expend=False):
    """
    Compute cross (or auto) fisher matrix Fll = Trace[invCAA.Pl.invCBB.Pl] = Trace[El.Pl]

    Parameters
    ----------
    El : ???
        ???

    Returns
    ----------
    ???
    """
    lmax = len(Pl) * 2**int(expend)
    lrange = np.arange(lmax)
    if expend:
        # pas de transpose car symm
        FAB = np.array([np.sum(El[il] * Expd(Pl,jl)) for il in lrange for jl in lrange]).reshape(lmax, lmax)
    else:
        # pas de transpose car symm
        FAB = np.array([np.sum(El[il] * Pl[jl]) for il in lrange for jl in lrange]).reshape(lmax, lmax)
    return FAB

def CrossWindowFunction(El, Pl):
    """
    Compute Tegmark cross (or auto) window matrix Wll = Trace[invCAA.Pl.invCBB.Pl] = Trace[El.Pl]
    Equivalent to Fisher matrix Fll when Tegmark B-matrix = 1

    Parameters
    ----------
    El : ???
        ???

    Returns
    ----------
    ???
    """
    lmax = len(El)
    lrange = np.arange((lmax))
    # pas de transpose car symm
    Wll = np.array([np.sum(El[il] * Pl[jl]) for il in lrange for jl in lrange]).reshape(lmax, lmax)
    return Wll

def CrossWindowFunctionLong(invCAA, invCBB, Pl):
    """
    Compute Tegmark cross (or auto) window matrix Wll = Trace[invCAA.Pl.invCBB.Pl] = Trace[El.Pl]
    Equivalent to Fisher matrix Fll when Tegmark B-matrix = 1

    Parameters
    ----------
    invCAA : ???
        ???

    Returns
    ----------
    ???
    """
    lmax = len(Pl)
    lrange = np.arange((lmax))
    # Pas de transpose car symm
    Wll = np.array([np.sum(np.dot(np.dot(invCAA, Pl[il]), invCBB) * Pl[jl]) for il in lrange for jl in lrange]).reshape(lmax, lmax)
    return Wll

def CrossMisherMatrix(El, CAA, CBB):
    """
    Compute matrix MAB = Trace[El.CAA.El.CBB] (see paper for definition)

    Parameters
    ----------
    El : ???
        ???

    Returns
    ----------
    ???
    """
    lmax = len(El)
    lrange = np.arange(lmax)
    El_CAA = np.array([np.dot(CAA,El[il]) for il in lrange])
    El_CBB = np.array([np.dot(CBB,El[il]) for il in lrange])
    MAB = np.array([np.sum(El_CAA[il] * El_CBB[jl].T) for il in lrange for jl in lrange]).reshape(lmax, lmax)
    return MAB

def CrossGisherMatrix(El, CAB):
    """
    Compute matrix GAB = Trace[El.CAB.El.CAB] (see paper for definition)

    Parameters
    ----------
    El : ???
        ???

    Returns
    ----------
    ???
    """
    lmax = len(El)
    lrange = np.arange(lmax)
    El_CAB = np.array([np.dot(CAB,El[il]) for il in lrange])
    GAB = np.array([np.sum(El_CAB[il] * El_CAB[jl].T) for il in lrange for jl in lrange]).reshape(lmax, lmax)
    return GAB

def CrossGisherMatrixLong(El, CAB):
    """
    Compute matrix GAB = Trace[El.CAB.El.CAB] (see paper for definition)

    Parameters
    ----------
    El : ???
        ???

    Returns
    ----------
    ???
    """
    lmax = len(El)
    lrange = np.arange(lmax)
    GAB = np.array([np.sum(np.dot(CAB, El[il]) * np.dot(CAB,El[jl]).T) for il in lrange for jl in lrange]).reshape(lmax, lmax)
    return GAB

def yQuadEstimator(dA, dB, El):
    """
    Compute pre-estimator 'y' such that Cl = Fll^-1 . yl

    Parameters
    ----------
    dA : ???
        ???

    Returns
    ----------
    ???
    """
    npix = len(dA)
    lrange = np.arange((len(El)))
    y = np.array([dA.dot(El[l]).dot(dB) for l in lrange])
    return y

def ClQuadEstimator(invW, y):
    """
    Compute estimator 'Cl' such that Cl = Fll^-1 . yl

    Parameters
    ----------
    invW : ???
        ???

    Returns
    ----------
    ???
    """
    Cl = np.dot(invW, y)
    return Cl

def biasQuadEstimator(NoiseN, El):
    """
    Compute bias term bl such that Cl = Fll^-1 . ( yl + bias)

    Parameters
    ----------
    NoiseN : ???
        ???

    Returns
    ----------
    ???
    """
    lrange = np.arange((len(El)))
    return np.array([np.sum(NoiseN * El[l]) for l in lrange])

def blEstimatorFlat(NoiseN, El):
    """
    Compute bias term bl such that Cl = Fll^-1 . ( yl + bl)
    Not to be confonded with beam function bl(fwhm)

    Parameters
    ----------
    NoiseN : ???
        ???

    Returns
    ----------
    ???
    """
    lrange = np.arange((len(El)))

    return np.array([np.sum(NoiseN * np.diag(El[l])) for l in lrange])


def CovAB(invWll, GAB):
    """
    Compute analytical covariance matrix Cov(Cl, Cl_prime)

    Parameters
    ----------
    invWll : ???
        ???

    Returns
    ----------
    ???
    """
    covAB = np.dot(np.dot(invWll, GAB), invWll.T) + invWll
    return covAB
