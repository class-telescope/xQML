"""
Set of routines to generate basic simulations

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
    muKarcmin : float
        Pixel noise [muK . arcmin]
    nside : int
        Healpix map resolution (power of 2)

    Returns
    ----------
    varperpix : float
        Variance per pixel

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
    pixvar : float
        Variance per pixel
    nside : int
        Healpix map resolution (power of 2)

    Returns
    ----------
    nl : float
        Noise spectrum level

    Example
    ----------
    >>> nside = 16
    >>> pixvar = muKarcmin2var(10.0, nside)
    >>> nl = pixvar2nl(pixvar, nside)
    >>> print(round(nl * 1e18))
    8.0
    """
    nl = pixvar*4.*np.pi/(12*nside**2.)
    return nl


def getNl(pixvar, nside, nbins):
    """
    Return noise spectrum for a given nside and pixel variance

    Parameters
    ----------
    pixvar : float
        Variance per pixel
    nside : int
        Healpix map resolution (power of 2)
    nbins : int
        Number of bins

    Returns
    ----------
    Nl : 1D array of float
        Noise spectrum

    Example
    ----------
    >>> nside = 16
    >>> pixvar = muKarcmin2var(10.0, nside)
    >>> nl = getNl(pixvar, nside, 2)
    >>> print(round(nl[0] * 1e18))
    8.0
    """
    Nl = pixvar*4.*np.pi/(12*nside**2.)*np.ones((nbins))
    return Nl


def getstokes(spec=None, temp=False, polar=False, corr=False):
    """
    Get the Stokes parameters number and name(s)

    Parameters
    ----------
    spec : bool
        If True, get Stokes parameters for polar (default: True)
    polar : bool
        If True, get Stokes parameters for polar (default: True)
    temp : bool
        If True, get Stokes parameters for temperature (default: False)
    corr : bool
        If True, get Stokes parameters for EB and TB (default: False)

    Returns
    ----------
    stokes : list of string
        Stokes variables names
    spec : int
        Spectra names
    istokes : list
        Indexes of power spectra

    Example
    ----------
    >>> getstokes(polar=True, temp=False, corr=False)
    (['Q', 'U'], ['EE', 'BB'], [1, 2])
    >>> getstokes(polar=True, temp=True, corr=False)
    (['I', 'Q', 'U'], ['TT', 'EE', 'BB', 'TE'], [0, 1, 2, 3])
    >>> getstokes(polar=True, temp=True, corr=True)
    (['I', 'Q', 'U'], ['TT', 'EE', 'BB', 'TE', 'EB', 'TB'], [0, 1, 2, 3, 4, 5])
    """
    if spec is not None:
        _temp = "TT" in spec or "TE" in spec or "TB" in spec or temp
        _polar = "EE" in spec or "BB" in spec or "TE" in spec or "TB" in \
            spec or "EB" in spec or polar
        _corr = "TE" in spec or "TB" in spec or "EB" in spec or corr
        if not _temp and not _polar and not _corr:
            print("invalid spectra list and/or no options")
    else:
        _temp = temp
        _polar = polar
        _corr = corr

    speclist = []
    if _temp or (spec is None and corr):
        speclist.extend(["TT"])
    if _polar:
        speclist.extend(["EE", "BB"])
    if spec is not None and not corr:
        if 'TE' in spec:
            speclist.extend(["TE"])
        if 'EB' in spec:
            speclist.extend(["EB"])
        if 'TB' in spec:
            speclist.extend(["TB"])

    elif _corr:
        speclist.extend(["TE", "EB", "TB"])

    stokes = []
    if _temp:
        stokes.extend(["I"])
    if _polar:
        stokes.extend(["Q", "U"])

    ispecs = [['TT', 'EE', 'BB', 'TE', 'EB', 'TB'].index(s) for s in speclist]
    istokes = [['I', 'Q', 'U'].index(s) for s in stokes]
    return stokes, speclist, istokes, ispecs


def GetBinningMatrix(
        ellbins, lmax, norm=False, polar=True,
        temp=False, corr=False):
    """
    Return P (m,n) and Q (n,m) binning matrices such that
    Cb = P.Cl and Vbb = P.Vll.Q with m the number of bins and
    n the number of multipoles.
    In addition, returns ell (total non-binned multipole range)
    and ellval (binned multipole range)

    Parameters
    ----------
    ellbins : list of integers
        Bins lower bound
    lmax : int
        Maximum multipole
    norm : bool (default: False)
        If True, weight the binning scheme such that P = l*(l+1)/(2*pi)
    polar : bool
        If True, get Stokes parameters for polar (default: True)
    temp : bool
        If True, get Stokes parameters for temperature (default: False)
    corr : bool
        If True, get Stokes parameters for EB and TB (default: False)

    Returns
    ----------
    P : array of float (m,n)
        Binning matrix such that Cb = P.Cl
    Q : array of int (n,m)
        Binning matrix such that P.Q = I
        or P.Q = I * l(l+1)/(2pi) if norm=True
    ell : array of int (n)
        Multipoles range
    ellvall : array of float (m)
        Bins pivot range

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
    # ### define Stokes
    nspec = 1

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

    allmasklm = nspec*[list(masklm)]
    masklM = np.array(sparse.block_diag(allmasklm[:]).toarray())
    binsnorm = np.array(
        nspec * [list(np.arange(minell[0], np.max(ellbins)))]).flatten()

    binsnorm = binsnorm*(binsnorm+1)/2./np.pi
    P = np.array(masklM)*1.
    Q = P.T
    P = P / np.sum(P, 1)[:, None]
    if norm:
        P *= binsnorm

    return P, Q, ell, ellval


def extrapolpixwin(nside, Slmax, pixwin=True):
    '''
    Parameters
    ----------
    nside : int
        Healpix map resolution
    Slmax : int
        Maximum multipole value computed for the pixel covariance pixel matrix
    pixwin : bool
        If True, multiplies the beam window function by the pixel
        window function. Default: True

    Returns
    ----------
    fpixwin : array of floats

    Example :
    ----------
    >>> print(hp.pixwin(2))
    [ 1.          0.977303    0.93310702  0.86971852  0.79038278  0.69905215
      0.60011811  0.49813949  0.39760902]
    >>> print(extrapolpixwin(2, 20, True))
    [ 1.          0.977303    0.93310702  0.86971852  0.79038278  0.69905215
      0.60011811  0.49813949  0.39760902  0.30702636  0.22743277  0.16147253
      0.10961864  0.07098755  0.04374858  0.02559774  0.01418623  0.00742903
      0.00366749  0.00170274]
    >>> print(extrapolpixwin(2, 20, False))
    [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
      1.  1.]
    '''
    if pixwin:
        prepixwin = np.array(hp.pixwin(nside))
        poly = np.polyfit(np.arange(len(prepixwin)), np.log(prepixwin),
                          deg=3, w=np.sqrt(prepixwin))
        y_int = np.polyval(poly, np.arange(Slmax))
        fpixwin = np.exp(y_int)
        fpixwin = np.append(prepixwin, fpixwin[len(prepixwin):])[: Slmax]
    else:
        fpixwin = np.ones((Slmax))

    return fpixwin


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
