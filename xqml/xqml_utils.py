"""
Set of routines to ...

Author: Vanneste
"""
from __future__ import division

import sys
import timeit

import numpy as np
from scipy import linalg, sparse


def pd_inv(a):
    n = a.shape[0]
    I = np.identity(n)
    return linalg.solve(a, I, sym_pos = True, overwrite_b = True)


def getstokes(spec):
    """
    Get the Stokes parameters number and name(s)

    Parameters
    ----------
    spec : bool
        If True, get Stokes parameters for polar (default: True)

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
    >>> getstokes(['EE','BB'])
    (['Q', 'U'], ['EE', 'BB'], [1, 2])
    >>> getstokes(['TT','EE','BB','TE'])
    (['I', 'Q', 'U'], ['TT', 'EE', 'BB', 'TE'], [0, 1, 2, 3])
    >>> getstokes(['TT', 'EE', 'BB', 'TE', 'EB', 'TB'])
    (['I', 'Q', 'U'], ['TT', 'EE', 'BB', 'TE', 'EB', 'TB'], [0, 1, 2, 3, 4, 5])
    """
    _temp  = "TT" in spec or "TE" in spec or "TB" in spec
    _polar = "EE" in spec or "BB" in spec or "TE" in spec or "TB" in spec or "EB" in spec
    _corr  = "TE" in spec or "TB" in spec or "EB" in spec
    if not _temp and not _polar and not _corr:
        print("invalid spectra list and/or no options")
    
    stokes = []
    if _temp:
        stokes.extend(["I"])
    if _polar:
        stokes.extend(["Q", "U"])
    
    ispecs = [['TT', 'EE', 'BB', 'TE', 'TB', 'EB'].index(s) for s in spec]
    istokes = [['I', 'Q', 'U'].index(s) for s in stokes]
    return stokes, spec, istokes, ispecs


def ComputeSizeDs_dcb(nside, fsky, deltal=1, unit="Gb"):
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
    if units[0].lower() == "g":
        toGB = 1024. * 1024. * 1024.
    else:
        toGB = 1.

    nspec = 2
    nell = nspec * (3.*nside-1) /deltal
    npixtot = 2*12*nside**2*fsky
    sizeds_dcb = (npixtot)**2 * (nell + 5)

    return( 8.* sizeds_dcb / toGB)
#    print("size (Gb) = " + str(sizeds_dcb))


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


def progress_bar(i, n):
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
    ntot = 50
    ndone = ntot * (i+1) / n
    a = '\r|'
    for k in np.arange(ndone):
        a += '#'
    for k in np.arange(ntot-ndone):
        a += ' '
    fra = (i+1.)/n
    a += '| %i %%' % round(fra*100.)
    sys.stdout.write(a)
    # sys.stdout.flush()
    if i == n-1:
        sys.stdout.write('\n')
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



def GetBinningMatrix( ellbins, lmax, norm=False):
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






class symarray(np.ndarray):

    # wrapper class for numpy array for symmetric matrices. New attribute can pack matrix to optimize storage.
    # Usage:
    # If you have a symmetric matrix A as a shape (n,n) numpy ndarray, Sym(A).packed is a shape (n(n+1)/2,) numpy array 
    # that is a packed version of A.  To convert it back, just wrap the flat list in Sym().  Note that Sym(Sym(A).packed)

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)

        if len(obj.shape) == 1:
            l = obj.copy()
            p = obj.copy()
#            m = int((np.sqrt(8 * len(obj) + 1) - 1) / 2)
#            sqrt_m = np.sqrt(m)
            m = (np.sqrt(8 * len(obj) + 1) - 1) / 2

#            if np.isclose(sqrt_m, np.round(sqrt_m)):
            if np.isclose(m, np.round(m)):
                m = int(m)
            else:
                raise ValueError('One dimensional input length must be a triangular number.')
            
            A = np.zeros((m, m))
            for i in range(m):
                A[i, i:] = l[:(m-i)]
                A[i:, i] = l[:(m-i)]
                l = l[(m-i):]
            obj = np.asarray(A).view(cls)
            obj.packed = p
        
        
        elif len(obj.shape) == 2:
            if obj.shape[0] != obj.shape[1]:
                raise ValueError('Two dimensional input must be a square matrix.')
            packed_out = []
            for i in range(obj.shape[0]):
                packed_out.append(obj[i, i:])
            obj.packed = np.concatenate(packed_out)
        
        else:
            raise ValueError('Input array must be 1 or 2 dimensional.')
        
        return obj
    
    def __array_finalize__(self, obj):
        if obj is None: return
        self.packed = getattr(obj, 'packed', None)


#n=5
#riri = arange( n*(n+1)/2)
#fifi = Sym(riri)
#fifi.packed
#
#riri = arange( n*n).reshape((n,n))
#riri = riri + riri.T
#fifi = Sym(riri)
#fifi.packed
