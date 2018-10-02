"""
Set of routines to ...

Author: Vanneste
"""
from __future__ import division

import sys
import timeit

import numpy as np


def ComputeSizeDs_dcb(nside, fsky, deltal=1):
    """
    ???

    Parameters
    ----------
    nside : ???
        ???

    Returns
    ----------
    ???
    """
    toGB = 1024. * 1024. * 1024.
    sizeds_dcb = (2*12*nside**2*fsky)**2*8*2*(3.*nside/deltal) / toGB
    print("size (Gb) = " + str(sizeds_dcb))
    print("possible reduced size (Gb) = " + str(sizeds_dcb/4))


def get_colors(num_colors):
    """
    ???

    Parameters
    ----------
    num_colors : ???
        ???

    Returns
    ----------
    ???
    """
    import colorsys
    colors = []
    ndeg = 250.
    for i in np.arange(0., ndeg, ndeg / num_colors):
        hue = i/360.
        lightness = 0.5
        saturation = 0.7
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return np.array(colors)


def progress_bar(i, n, dt):
    """
    ???

    Parameters
    ----------
    i : ???
        ???

    Returns
    ----------
    ???
    """
    if n != 1:
        ntot = 50
        ndone = ntot * i / (n - 1)
        a = '\r|'
        for k in np.arange(ndone):
            a += '#'
        for k in np.arange(ntot-ndone):
            a += ' '
        fra = i/(n-1.)
        remain = dt/fra*(1-fra)
        minu = remain/60.
        a += '| %i %% : %.1f sec (%.1f min)' % (int(100.*fra), remain, minu)
        sys.stdout.write(a)
        # sys.stdout.flush()
        if i == n-1:
            sys.stdout.write(
                '\n => Done, total time = %.1f sec (%.1f min)\n' % (dt, dt/60.))
            sys.stdout.flush()


def check_symmetric(a, tol=1e-8):
    """
    ???

    Parameters
    ----------
    a : ???
        ???

    Returns
    ----------
    ???
    """
    return np.allclose(a, a.T, atol=tol)


def randomword(length):
    """
    ???

    Parameters
    ----------
    length : ???
        ???

    Returns
    ----------
    ???
    """
    return ''.join(rd.choice(string.lowercase) for i in range(length))


def cov_from_maps(maps0, maps1):
    """
    ???

    Parameters
    ----------
    maps0 : ???
        ???

    Returns
    ----------
    ???
    """
    sh = np.shape(maps0)
    npix = sh[1]
    nbmc = sh[0]
    covmc = np.zeros((npix, npix))
    mm0 = np.mean(maps0, axis=0)
    # print(mm0)
    mm1 = np.mean(maps1, axis=0)
    # print(mm1)
    themaps0 = np.zeros((nbmc, npix))
    themaps1 = np.zeros((nbmc, npix))
    start = timeit.default_timer()
    for i in np.arange(npix):
        progress_bar(i, npix, timeit.default_timer()-start)
        themaps0[:, i] = maps0[:, i] - mm0[i]
        themaps1[:, i] = maps1[:, i] - mm1[i]
    print('hy')
    start = timeit.default_timer()
    for i in np.arange(npix):
        progress_bar(i, npix, timeit.default_timer()-start)
        for j in np.arange(npix):
            covmc[i, j] = np.mean(themaps0[:, i] * themaps1[:, j])
    return(covmc)


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



# def GetBinningMatrix(
#         ellbins, lmax, norm=False, polar=True,
#         temp=False, corr=False):
#     """
#     Return P (m,n) and Q (n,m) binning matrices such that
#     Cb = P.Cl and Vbb = P.Vll.Q with m the number of bins and
#     n the number of multipoles.
#     In addition, returns ell (total non-binned multipole range)
#     and ellval (binned multipole range)

#     Parameters
#     ----------
#     ellbins : list of integers
#         Bins lower bound
#     lmax : int
#         Maximum multipole
#     norm : bool (default: False)
#         If True, weight the binning scheme such that P = l*(l+1)/(2*pi)
#     polar : bool
#         If True, get Stokes parameters for polar (default: True)
#     temp : bool
#         If True, get Stokes parameters for temperature (default: False)
#     corr : bool
#         If True, get Stokes parameters for EB and TB (default: False)

#     Returns
#     ----------
#     P : array of float (m,n)
#         Binning matrix such that Cb = P.Cl
#     Q : array of int (n,m)
#         Binning matrix such that P.Q = I
#         or P.Q = I * l(l+1)/(2pi) if norm=True
#     ell : array of int (n)
#         Multipoles range
#     ellvall : array of float (m)
#         Bins pivot range

#     Example
#     ----------
#     >>> bins = np.array([2.0, 5.0, 10.0])
#     >>> P, Q, ell, ellval = GetBinningMatrix(
#     ...     bins, 10.0)
#     >>> print(P) # doctest: +NORMALIZE_WHITESPACE
#     [[ 0.33333333  0.33333333  0.33333333  0.          0.          0.       0.
#        0.          0.          0.          0.          0.          0.       0.
#        0.          0.          0.          0.        ]
#      [ 0.          0.          0.          0.2         0.2         0.2      0.2
#        0.2         0.          0.          0.          0.          0.       0.
#        0.          0.          0.          0.        ]
#      [ 0.          0.          0.          0.          0.          0.       0.
#        0.          0.          0.33333333  0.33333333  0.33333333  0.       0.
#        0.          0.          0.          0.        ]
#      [ 0.          0.          0.          0.          0.          0.       0.
#        0.          0.          0.          0.          0.          0.2      0.2
#        0.2         0.2         0.2         0.        ]]
#     >>> print(Q) # doctest: +NORMALIZE_WHITESPACE
#     [[ 1.  0.  0.  0.]
#      [ 1.  0.  0.  0.]
#      [ 1.  0.  0.  0.]
#      [ 0.  1.  0.  0.]
#      [ 0.  1.  0.  0.]
#      [ 0.  1.  0.  0.]
#      [ 0.  1.  0.  0.]
#      [ 0.  1.  0.  0.]
#      [ 0.  0.  0.  0.]
#      [ 0.  0.  1.  0.]
#      [ 0.  0.  1.  0.]
#      [ 0.  0.  1.  0.]
#      [ 0.  0.  0.  1.]
#      [ 0.  0.  0.  1.]
#      [ 0.  0.  0.  1.]
#      [ 0.  0.  0.  1.]
#      [ 0.  0.  0.  1.]
#      [ 0.  0.  0.  0.]]
#     >>> print(ell)
#     [  2.   3.   4.   5.   6.   7.   8.   9.  10.  11.]
#     >>> print(ellval)
#     [ 3.  7.]
#     """
#     # ### define Stokes
#     stokes, spec, istokes, ispecs = getstokes(polar=polar, temp=temp, corr=corr)
#     nspec = len(spec)

#     nbins = len(ellbins) - 1
#     ellmin = np.array(ellbins[0: nbins])
#     ellmax = np.array(ellbins[1: nbins + 1]) - 1
#     ell = np.arange(np.min(ellbins), lmax + 2)
#     maskl = (ell[:-1] < (lmax + 2)) & (ell[:-1] > 1)

#     # define min
#     minell = np.array(ellbins[0: nbins])
#     # and max of a bin
#     maxell = np.array(ellbins[1: nbins + 1]) - 1
#     ellval = (minell + maxell) * 0.5

#     masklm = []
#     for i in np.arange(nbins):
#         masklm.append(((ell[:-1] >= minell[i]) & (ell[:-1] <= maxell[i])))

#     allmasklm = nspec*[list(masklm)]
#     masklM = np.array(sparse.block_diag(allmasklm[:]).toarray())
#     binsnorm = np.array(
#         nspec * [list(np.arange(minell[0], np.max(ellbins)))]).flatten()

#     binsnorm = binsnorm*(binsnorm+1)/2./np.pi
#     P = np.array(masklM)*1.
#     Q = P.T
#     P = P / np.sum(P, 1)[:, None]
#     if norm:
#         P *= binsnorm

#     return P, Q, ell, ellval
