"""
Utils for xQML code
"""
from __future__ import division

import sys
import timeit

import numpy as np
from scipy import linalg, sparse


def pd_inv(a):
    '''
    Quicker to inverse for pixel-pixel matrices
    (almost 5x quicker)
    '''
    n = a.shape[0]
    I = np.identity(n)
    # inv = linalg.solve(a, I, sym_pos=True, overwrite_b=True, )
    inv = linalg.solve(a, I, assume_a='sym', overwrite_b=True, )
    return np.ascontiguousarray(inv)


def getstokes(spec):
    """
    Get the Stokes parameters number and name(s)

    Parameters
    ----------
    spec : tuple, list
        If True, get Stokes parameters for polar (default: True)

    Returns
    ----------
    stokes : list of string
        Stokes variables names
    spec : int
        Spectra names
    istokes : list
        Indexes of stokes parameter in ['T', 'Q', 'U']
    ispecs: list of int
        Index of power spectra in ['TT', 'EE', 'BB', 'TE', 'TB', 'EB']
    Example
    ----------
    >>> getstokes(['EE','BB'])
    (['Q', 'U'], ['EE', 'BB'], [1, 2])
    >>> getstokes(['TT','EE','BB','TE'])
    (['I', 'Q', 'U'], ['TT', 'EE', 'BB', 'TE'], [0, 1, 2, 3])
    >>> getstokes(['TT', 'EE', 'BB', 'TE', 'TB', 'EB'])
    (['I', 'Q', 'U'], ['TT', 'EE', 'BB', 'TE', 'TB', 'EB'], [0, 1, 2, 3, 4, 5])
    """
    _temp = "TT" in spec or "TE" in spec or "TB" in spec
    _polar = "EE" in spec or "BB" in spec or "TE" in spec or "TB" in spec or "EB" in spec
    _corr = "TE" in spec or "TB" in spec or "EB" in spec
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


def progress_bar(it,ntot,name=""):
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(it, int):
        progress = float(it+1)/ntot
    else:
        progress = it
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\r{0} [{1}] {2}% {3}".format( name, "#"*block + "-"*(barLength-block), round(progress*100), status)
    sys.stdout.write(text)
    sys.stdout.flush()


def GetBinningMatrix(ellbins, lmax, norm=False):
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
    binsnorm = np.array(nspec * [list(np.arange(minell[0], np.max(ellbins)))]).flatten()

    binsnorm = binsnorm*(binsnorm+1)/2./np.pi
    P = np.array(masklM)*1.
    Q = P.T
    P = P / np.sum(P, 1)[:, None]
    if norm:
        P *= binsnorm

    return P, Q, ell, ellval


def tri2upper(m):
    """
    expand packed triangular matrix to upper matrix of shape (npix, npix).
    The lower triangle is filled with zeros.
    Parameters
    ----------
    m: np.ndarray
        shape (..., n_tri)
    Returns
    -------
    np.ndarray, shape (..., n_pix, n_pix)
    """
    n_tri = m.shape[-1]
    n = int(np.sqrt(8*n_tri+1)-1)//2
    out = np.zeros((*m.shape[:-1], n, n), dtype=m.dtype)
    i, j = np.triu_indices(n=n, )
    out[..., i, j] =m[..., :]
    return out


def upper2tri(m):
    """
    Packed symmetric matrix to triangular form.
    Parameters
    ----------
    m: np.ndarray
        shape (..., n_tri)
    Returns
    -------
    np.ndarray, shape (..., n_pix, n_pix)
    """
    n = m.shape[-1]
    n_tri = (n*(n+1))//2
    i, j = np.triu_indices(n=n, )
    out =np.zeros((*m.shape[:-2], n_tri), dtype=m.dtype)
    out[..., :] = m[..., i, j]
    return out


def Cl4to6(Cl):
    """
    Zero-pad the Cl spectra in shape (4, nl) to (6, nl)
    Parameters
    ----------
    Cl: np.ndarray
        spectra in the order of TT/EE/BB/TE
    Returns
    -------
    np.ndarray
        spectra in the order of TT/EE/BB/TE/TB/EB
    """
    _Cl = np.array(Cl)
    if _Cl.shape[0] == 4:
        return np.pad(_Cl, ((0, 2), (0, 0)), constant_values=0)
    elif _Cl.shape[0] == 6:
        return _Cl
    else:
        raise ValueError(f"Wrong spectra shape: {_Cl.shape}")


class symarray(np.ndarray):
    '''
    wrapper class for numpy array for symmetric matrices. New attribute can pack matrix to optimize storage.
    Usage:
    If you have a symmetric matrix A as a shape (n,n) numpy ndarray, Sym(A).packed is a shape (n(n+1)/2,) numpy array 
    that is a packed version of A.  To convert it back, just wrap the flat list in Sym().  Note that Sym(Sym(A).packed)
    '''
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
        if obj is None:
            return
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
