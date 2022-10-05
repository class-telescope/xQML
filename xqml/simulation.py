"""
Set of routines to generate basic simulations

Author: Vanneste
"""
from __future__ import division

import numpy as np
import healpy as hp



def Karcmin2var(Karcmin, nside):
    """
    Return pixel variance for a given nside and noise level [K . arcmin]

    Parameters
    ----------
    Karcmin : float
        Pixel noise [K . arcmin]
    nside : int
        Healpix map resolution (power of 2)

    Returns
    ----------
    varperpix : float
        Variance per pixel

    Example
    ----------
    >>> var = Karcmin2var(10e-6, 16)
    >>> print(round(var * 1e16))
    21.0
    """
    pixarea = hp.nside2pixarea(nside, degrees=True)
    varperpix = (Karcmin / 60.)**2 / pixarea
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
    >>> pixvar = Karcmin2var(10e-6, nside)
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
    >>> pixvar = Karcmin2var(10e-6, nside)
    >>> nl = getNl(pixvar, nside, 2)
    >>> print(round(nl[0] * 1e18))
    8.0
    """
    Nl = pixvar*4.*np.pi/(12*nside**2.)*np.ones((nbins))
    return Nl



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
    [  1.00000000e+00   9.77303000e-01   9.33107017e-01   8.69718524e-01
       7.90382779e-01   6.99052151e-01   6.00118114e-01   4.98139486e-01
       3.97609016e-01   3.07026358e-01   2.27432772e-01   1.61472532e-01
       1.09618639e-01   7.09875545e-02   4.37485835e-02   2.55977424e-02
       1.41862346e-02   7.42903370e-03   3.66749182e-03   1.70274467e-03
       7.41729191e-04]
    >>> print(extrapolpixwin(2, 20, False))
    [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.  1.
      1.  1.  1.]
    '''
    if pixwin:
        prepixwin = np.array(hp.pixwin(nside))
        poly = np.polyfit(np.arange(len(prepixwin)), np.log(prepixwin),
                          deg=3, w=np.sqrt(prepixwin))
        y_int = np.polyval(poly, np.arange(Slmax+1))
        fpixwin = np.exp(y_int)
        fpixwin = np.append(prepixwin, fpixwin[len(prepixwin):])[:Slmax+1]
    else:
        fpixwin = np.ones((Slmax+1))

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
