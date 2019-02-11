"""
Set of routines to ...

Author: Vanneste
"""
from __future__ import division

import numpy as np


def Pl(ds_dcb):
    """
    Reshape ds_dcb (nspec, nbins) into Pl (nspec * nbins)

    Parameters
    ----------
    ds_dcb : ndarray of floats
        Normalize Legendre polynomials (2l + 1)/2pi * pl

    Returns
    ----------
    Pl : ndarray of floats
        Rescaled normalize Legendre polynomials dS/dCl

    Example
    ----------
    >>> thePl = Pl(np.arange(16).reshape((2,2,2,2)))
    >>> print(thePl) # doctest: +NORMALIZE_WHITESPACE
    [[[ 0  1]
      [ 2  3]]
    <BLANKLINE>
     [[ 4  5]
      [ 6  7]]
    <BLANKLINE>
     [[ 8  9]
      [10 11]]
    <BLANKLINE>
     [[12 13]
      [14 15]]]
    """
    nnpix = np.shape(ds_dcb)[-1]
    return np.copy(ds_dcb).reshape(2 * (np.shape(ds_dcb)[1]), nnpix, nnpix)


def CorrelationMatrix(Clth, Pl, ellbins, polar=True, temp=False, corr=False):
    """
    Compute correlation matrix S = sum_l Pl*Cl

    Parameters
    ----------
    Clth : 1D array of float
        Fiducial spectra
    Pl : ndarray of floats
        Rescaled normalize Legendre polynomials dS_dCb
    ellbins : array of integers
        Bins lower bound
    polar : bool
        If True, get Stokes parameters for polar (default: True)
    temp : bool
        If True, get Stokes parameters for temperature (default: False)
    corr : bool
        If True, get Stokes parameters for EB and TB (default: False)

    Returns
    ----------
    S : 2D square matrix of float (npix, npix)
        Pixel covariance matrix

    Example
    ----------
    >>> Pl = np.arange(10).reshape(2,-1)
    >>> Clth = np.arange(40).reshape(4,-1)
    >>> ellbins = np.arange(2,10,1)
    >>> S = CorrelationMatrix(Clth, Pl, ellbins)
    >>> print(S) # doctest: +NORMALIZE_WHITESPACE
    [[   0  280  560  840 1120]
     [1400 1680 1960 2240 2520]]
    """
    if corr:
        xx = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
        ind = [0, 1, 2, 3, 4, 5]
    else:
        xx = ['TT', 'EE', 'BB', 'TE']
        ind = [0, 1, 2, 3]

    if not temp:
        allStoke = ['Q', 'U']
        if corr:
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
    S = np.sum(Pl * clth[:, None, None], 0)
    return S


def El(invCAA, invCBB, Pl):
    """
    Compute El = CAA^-1.Pl.CBB^-1

    Parameters
    ----------
    invCAA : square matrix array of float
        Inverse pixel covariance matrix of dataset A
    invCBB : square matrix array of float
        Inverse pixel covariance matrix of dataset B
    Pl : ndarray of floats
        Rescaled normalize Legendre polynomials dS/dCl

    Returns
    ----------
    El : array of float (shape(Pl))
        Quadratic parameter matrices such that yl = dA.El.dB.T

    Example
    ----------
    >>> Pl = np.arange(12).reshape(3,2,2)
    >>> invCAA = np.array([[1,2],[2,3]])
    >>> invCBB = np.array([[4,3],[3,6]])
    >>> print(El(invCAA, invCBB, Pl))
    [[[ 37  54]
      [ 57  84]]
    <BLANKLINE>
     [[121 162]
      [197 264]]
    <BLANKLINE>
     [[205 270]
      [337 444]]]

    """

    lmax = len(Pl)
    lrange = np.arange(lmax)
    npix = len(invCAA)
    # El = np.array([np.dot(np.dot(invCAA, Pl[l]), invCBB) for l in lrange]).reshape((lmax, npix, npix))
    El = np.array([np.linalg.multi_dot((invCAA, Pl[l], invCBB)) for l in lrange]).reshape((lmax, npix, npix))
    return El


def CrossWindowFunction(El, Pl):
    """
    Compute mode-mixing matrix (Tegmark's window matrix)
    Wll = Trace[invCAA.Pl.invCBB.Pl] = Trace[El.Pl]

    Parameters
    ----------
    El : ndarray of floats
        Quadratic parameter matrices such that yl = dA.El.dB.T
    Pl : ndarray of floats
        Rescaled normalize Legendre polynomials dS/dCl

    Returns
    ----------
    Wll : 2D square matrix array of floats
        Mode-mixing matrix of dimension (nspec * nbins)

    Example
    ----------
    >>> Pl = np.arange(12).reshape(3,2,2)
    >>> El = np.arange(12,24).reshape(3,2,2)
    >>> print(CrossWindowFunction(El, Pl))
    [[ 86 302 518]
     [110 390 670]
     [134 478 822]]
    """
    lmax = len(El)
    lrange = np.arange((lmax))
    # pas de transpose car symm
    Wll = np.array(
        [np.sum(El[il] * Pl[jl]) for il in lrange for jl in lrange]
        ).reshape(lmax, lmax)
    return Wll


def CrossWindowFunctionLong(invCAA, invCBB, Pl):
    """
    Compute mode-mixing matrix (Tegmark's window matrix)
    Wll = Trace[invCAA.Pl.invCBB.Pl] = Trace[El.Pl]

    Parameters
    ----------
    invCAA : square matrix array of float
        Inverse pixel covariance matrix of dataset A
    invCBB : square matrix array of float
        Inverse pixel covariance matrix of dataset B
    Pl : ndarray of floats
        Rescaled normalize Legendre polynomials dS/dCl

    Returns
    ----------
    Wll : 2D square matrix array of floats
        Mode-mixing matrix of dimension (nspec * nbins, nspec * nbins)

    Example
    ----------
    >>> Pl = np.arange(12).reshape(3,2,2)
    >>> invCAA = np.array([[1,2],[2,3]])
    >>> invCBB = np.array([[4,3],[3,6]])
    >>> print(CrossWindowFunctionLong(invCAA, invCBB, Pl))
    [[  420  1348  2276]
     [ 1348  4324  7300]
     [ 2276  7300 12324]]
    """
    lmax = len(Pl)
    lrange = np.arange((lmax))
    # Pas de transpose car symm
    Wll = np.array(
        [np.sum(np.dot(np.dot(invCAA, Pl[il]), invCBB) * Pl[jl])
         for il in lrange for jl in lrange]).reshape(lmax, lmax)
    return Wll


def CrossGisherMatrix(El, CAB):
    """
    Compute matrix GAB = Trace[El.CAB.El.CAB]

    Parameters
    ----------
    CAB : 2D square matrix array of floats
        Pixel covariance matrix between dataset A and B
    El : ndarray of floats
        Quadratic parameter matrices such that yl = dA.El.dB.T

    Returns
    ----------
    GAB : 2D square matrix array of floats

     Example
    ----------
    >>> El = np.arange(12).reshape(3,2,2)
    >>> CAB = np.array([[1,2],[2,3]])
    >>> print(CrossGisherMatrix(El, CAB))
    [[ 221  701 1181]
     [ 701 2205 3709]
     [1181 3709 6237]]
    """
    lmax = len(El)
    lrange = np.arange(lmax)
    El_CAB = np.array([np.dot(CAB, El[il]) for il in lrange])
    GAB = np.array(
        [np.sum(El_CAB[il] * El_CAB[jl].T) for il in lrange for jl in lrange]
        ).reshape(lmax, lmax)
    return GAB


def CrossGisherMatrixLong(El, CAB):
    """
    Compute matrix GAB = Trace[El.CAB.El.CAB]

    Parameters
    ----------
    CAB : 2D square matrix array of floats
        Pixel covariance matrix between dataset A and B
    El : ndarray of floats
        Quadratic parameter matrices such that yl = dA.El.dB.T

    Returns
    ----------
    GAB : 2D square matrix array of floats

    Example
    ----------
    >>> El = np.arange(12).reshape(3,2,2)
    >>> CAB = np.array([[1,2],[2,3]])
    >>> print(CrossGisherMatrixLong(El, CAB))
    [[ 221  701 1181]
     [ 701 2205 3709]
     [1181 3709 6237]]

    """
    lmax = len(El)
    lrange = np.arange(lmax)
    GAB = np.array(
        [np.sum(np.dot(CAB, El[il]) * np.dot(CAB, El[jl]).T)
         for il in lrange for jl in lrange]).reshape(lmax, lmax)
    return GAB


def yQuadEstimator(dA, dB, El):
    """
    Compute pre-estimator 'y' such that Cl = Fll^-1 . yl

    Parameters
    ----------
    dA : array of floats
        Pixels dataset A
    dB : array of floats
        Pixels dataset B
    El : ndarray of floats
        Quadratic parameter matrices such that yl = dA.El.dB.T

    Returns
    ----------
    >>> dA = np.arange(12)
    >>> dB = np.arange(12,24)
    >>> El = np.arange(3*12**2).reshape(3,12,12)
    >>> print(yQuadEstimator(dA, dB, El))
    [1360788 3356628 5352468]
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
    invW : 2D square matrix array of floats
        Inverse mode-mixing matrix Wll'^-1

    Returns
    ----------
    Cl : array of floats
        Unbiased estimated spectra

    Example
    ----------
    >>> invW = np.array([[1,2], [2,4]])
    >>> yl = np.array([3,7])
    >>> print(ClQuadEstimator(invW, yl))
    [17 34]
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


# def blEstimatorFlat(NoiseN, El):
#     """
#     Compute bias term bl such that Cl = Fll^-1 . ( yl + bl)
#     Not to be confonded with beam function bl(fwhm)

#     Parameters
#     ----------
#     NoiseN : ???
#         ???

#     Returns
#     ----------
#     ???
#     """
#     lrange = np.arange((len(El)))

#     return np.array([np.sum(NoiseN * np.diag(El[l])) for l in lrange])


def CovAB(invWll, GAB):
    """
    Compute analytical covariance matrix Cov(Cl, Cl_prime)

    Parameters
    ----------
    invWll : 2D square matrix of floats
        Inverse of the mode-mixing matrix Wll'=Tr[El.Pl']

    Returns
    ----------
    covAB : 2D square matrix array of floats
        Analytical covariance of estimate spectra

    Example
    ----------
    >>> invW = np.array([[1,2], [2,4]])
    >>> GAB = np.array([[5,3], [3,2]])
    >>> print(CovAB(invW, GAB))
    [[ 26  52]
     [ 52 104]]
    """
    covAB = np.dot(np.dot(invWll, GAB), invWll.T) + invWll
    return covAB


# def GetCorr(F):
#     """
#     Return correlation of covariance matrix
#     corr(F) = F[i, j] / sqrt(F[i, i]*F[j, j])

#     Parameters
#     ----------
#     F : 2D square matrix array (n,n)
#         Covariance matrix

#     Returns
#     ----------
#     Corr : 2D square matrix array (n,n)
#         Correlation matrix from F

#     Example
#     ----------
#     >>> CorrF = GetCorr(np.matrix('1 .5; .5 4'))
#     >>> print(CorrF) # doctest: +NORMALIZE_WHITESPACE
#     [[ 1.    0.25]
#      [ 0.25  1.  ]]
#     """
#     nbins = len(F)
#     Corr = np.array(
#         [F[i, j] / (F[i, i]*F[j, j])**.5 for i in np.arange(nbins)
#             for j in np.arange(nbins)]).reshape(nbins, nbins)
#     return Corr

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
