"""
Set of routines to ...

Author: Vanneste
"""
from __future__ import division

import timeit

import numpy as np
import healpy as hp
import math
from scipy import special

from .simulation import extrapolpixwin
from .xqml_utils import getstokes, progress_bar, symarray, GetBinningMatrix

import _libcov as clibcov


def compute_ds_dcb( ellbins, nside, ipok, bl, clth, Slmax, spec,
                    pixwin=False, timing=False, MC=0, Sonly=False, openMP=True, SymCompress=False):
    """
    Compute the Pl = dS/dCl matrices.

    Parameters
    ----------
    ellbins : array of floats
        Lowers bounds of bins. Example : taking ellbins = (2, 3, 5) will
        compute the spectra for bins = (2, 3.5).
    nside : int
        Healpix map resolution
    ipok : array of ints
        Healpy pixels numbers considered
    bl : 1D array of floats
        Beam window function
    clth : 4D or 6D array of float
        Fiducial power spectra
    Slmax : int
        Maximum lmax computed for the pixel covariance pixel matrix
    spec : 1D array of string
        Spectra list
    pixwin : bool
        If True, multiplies the beam window function by the pixel
        window function. Default: True
    timing : bool
        If True, displays timmer. Default: False
    MC : int
        If not 0, computes Pl using Monte-Carlo method from MC simulations.
        Default: False
    Sonly : bool
        If True, compute the signal matric only. Default: False


    Returns
    ----------
    Pl : ndarray of floats
        Normalize Legendre polynomials dS/dCl

    S :  2D array of floats
        Pixel signal covariance matrix S

    Example
    ----------
    >>> Pl, S = compute_ds_dcb(
    ... np.array([2,4,5,10]), 2, np.array([0,1,4,10,11]), np.arange(10),
    ... clth=np.arange(60).reshape(6,-1), Slmax=9,
    ... spec=["TT", "EE", "BB", "TE", "EB", "TB"],
    ... pixwin=True, timing=False, MC=0, Sonly=False)
    >>> print(round(np.sum(Pl),5), round(np.sum(S),5))
    (1149.18805, 23675.10206)
    """
    if Slmax < ellbins[-1]-1:
        print("WARNING : Slmax < lmax")

    # ### define pixels
    rpix = np.array(hp.pix2vec(nside, ipok))
    allcosang = np.dot(np.transpose(rpix), rpix)
    allcosang[allcosang > 1] = 1.0
    allcosang[allcosang < -1] = -1.0

    start = timeit.default_timer()

    temp = "TT" in spec
    polar = "EE" in spec or "BB" in spec
    corr = "TE" in spec or "TB" in spec or "EB" in spec
    if Sonly:
        if MC:
            S = S_bins_MC(
                ellbins, nside, ipok, allcosang, bl, clth, Slmax, MC,
                polar=polar, temp=temp, corr=corr,
                pixwin=pixwin, timing=timing)
        else:
            S = compute_S(
                ellbins, nside, ipok, allcosang, bl, clth, Slmax,
                polar=polar, temp=temp, corr=corr,
                pixwin=pixwin, timing=timing)
        return S

    if MC:
        Pl, S = covth_bins_MC(
            ellbins, nside, ipok, allcosang, bl, clth, Slmax, MC,
            polar=polar, temp=temp, corr=corr, pixwin=pixwin, timing=timing)
    elif openMP:
        fpixwin = extrapolpixwin(nside, Slmax, pixwin)
        bell = np.array([bl*fpixwin]*4)[:Slmax+1].ravel()
        stokes, spec, istokes, ispecs = getstokes(spec)
        nbins = (len(ellbins)-1)*len(spec)
        npix = len(ipok)*len(istokes)
        Pl = np.ndarray( nbins*npix**2)
        clibcov.dSdC( nside, len(istokes), ellbins, ipok, bell, Pl)
        Pl = Pl.reshape( nbins, npix, npix)
        P, Q, ell, ellval = GetBinningMatrix(ellbins, Slmax)
        S = SignalCovMatrix(Pl,np.array([P.dot(clth[isp,2:int(ellbins[-1])]) for isp in ispecs]).ravel())
    else:
        Pl, S = compute_PlS(
            ellbins, nside, ipok, allcosang, bl, clth, Slmax,
            spec=spec, pixwin=pixwin, timing=timing)

    #print( "Total time (npix=%d): %.1f sec" % (len(ipok),timeit.default_timer()-start))

    if SymCompress:
        Pl = np.array([symarray(P).packed for P in Pl])
        
    return Pl, S




def SignalCovMatrix(Pl, model, SymCompress=False):
    """
    Compute correlation matrix S = sum_l Pl*Cl
    
    Parameters
    ----------
    clth : ndarray of floats
    Array containing fiducial CMB spectra (unbinned).
    """
    # Return scalar product btw Pl and the fiducial spectra.
    return np.sum([symarray(P) for P in Pl] * model[:, None, None], 0)






def compute_PlS(
        ellbins, nside, ipok, allcosang, bl, clth, Slmax,
        spec, pixwin=True, timing=False):
    """
    Computes Legendre polynomes Pl = dS/dCb and signal matrix S.

    Parameters
    ----------
    ellbins : array of floats
        Lowers bounds of bins. Example : taking ellbins = (2, 3, 5) will
        compute the spectra for bins = (2, 3.5).
    nside : int
        Healpix map resolution
    ipok : array of ints
        Healpy pixels numbers considered
    bl : 1D array of floats
        Beam window function
    clth : 4D or 6D array of float
        Fiducial power spectra
    Slmax : int
        Maximum lmax computed for the pixel covariance pixel matrix
    spec : 1D array of string
        Spectra list
    pixwin : bool
        If True, multiplies the beam window function by the pixel
        window function. Default: True
    timing : bool
        If True, displays timmer. Default: False

    Returns
    ----------
    Pl : ndarray of floats
        Normalize Legendre polynomials dS/dCl

    S :  2D array of floats
        Pixel signal covariance matrix S

    Example
    ----------
    >>> Pl, S = compute_PlS(np.array([2,3,4,10]),4,ipok=np.array([0,1,3]),
    ... allcosang=np.linspace(0,1,15).reshape(3,-1), bl=np.arange(13),
    ... clth=np.arange(6*13).reshape(6,-1), Slmax=9,
    ... spec=["TT", "EE", "BB", "TE", "EB", "TB"], pixwin=True, timing=False)
    >>> print(round(np.sum(Pl),5), round(np.sum(S),5))
    (-429.8591, -17502.8982)

    >>> Pl, S = compute_PlS(np.array([2,3,4,10]),4,ipok=np.array([0,1,3]),
    ... allcosang=np.linspace(0,1,15).reshape(3,-1), bl=np.arange(13),
    ... clth=np.arange(6*13).reshape(6,-1), Slmax=9,
    ... spec=["TT", "EE", "BB", "TE", "EB", "TB"], pixwin=False, timing=False)
    >>> print(round(np.sum(Pl),5), round(np.sum(S),5))
    (-756.35517, -31333.69722)
    """

    lmax = ellbins[-1]
    ell = np.arange(np.min(ellbins), np.max(ellbins) + 1)
    nbins = len(ellbins) - 1
    minell = np.array(ellbins[0:nbins])
    maxell = np.array(ellbins[1:nbins+1]) - 1
    ellval = (minell + maxell) * 0.5

    npi = ipok.size
    stokes, spec, istokes, ispecs = getstokes(spec)
    nspec = len(spec)
    nsto = len(stokes)
    temp = "TT" in spec
    polar = "EE" in spec or "BB" in spec
    TE = 'TE' in spec
    EB = 'EB' in spec
    TB = 'TB' in spec
    te = spec.index('TE') if TE else 0
    tb = spec.index('TB') if TB else 0
    eb = spec.index('EB') if EB else 0
    ponbins = nbins*temp
    ponpi = npi*temp
    tenbins = te*nbins
    tbnbins = tb*nbins
    ebnbins = eb*nbins

    rpix = np.array(hp.pix2vec(nside, ipok))
    ll = np.arange(Slmax+1)
    fpixwin = extrapolpixwin(nside, Slmax, pixwin)
    norm = (2*ll[2:]+1)/(4.*np.pi)*(fpixwin[2:]**2)*(bl[2:Slmax+1]**2)
    clthn = clth[:, 2: Slmax+1]
    masks = []
    for i in np.arange(nbins):
        masks.append((ll[2:] >= minell[i]) & (ll[2:] <= maxell[i]))
    masks = np.array(masks)

    Pl = np.zeros((nspec*nbins, nsto*npi, nsto*npi))
    S = np.zeros((nsto*npi, nsto*npi))

    start = timeit.default_timer()

    for i in np.arange(npi):
        if timing:
            progress_bar(i, npi, -0.5 * (start-timeit.default_timer()))
        for j in np.arange(i, npi):
            cos_chi = allcosang[i, j]
            if temp:
                pl = norm*pl0(cos_chi, Slmax)[2:]
                elem = np.sum((pl * clthn[0]))
                S[i, j] = elem
                S[j, i] = elem
                for b in np.arange(nbins):
                    elem = np.sum(pl[masks[b]])
                    Pl[b, i, j] = elem
                    Pl[b, j, i] = elem

            if polar:
                ii = i+ponpi
                jj = j+ponpi
                cij, sij = polrotangle(rpix[:, i], rpix[:, j])
                cji, sji = polrotangle(rpix[:, j], rpix[:, i])

                # Tegmark version
                Q22 = norm * F1l2(cos_chi, Slmax)
                # # /!\ signe - !
                R22 = -norm * F2l2(cos_chi, Slmax)

                # # Matt version
                # d2p2 = dlss(cos_chi, 2,  2, Slmax)
                # d2m2 = dlss(cos_chi, 2, -2, Slmax)
                # Q22 = norm * ( d2p2 + d2m2 )[2:]/2.
                # R22 = norm * ( d2p2 - d2m2 )[2:]/2.

                if TE or TB:
                    # # Matt version
                    d20 = -dlss(cos_chi, 2,  0, Slmax)[2:]
                    P02 = norm * d20
                    # P02 = -norm * F1l0(cos_chi, Slmax)
                    elemA = 0
                    elemB = 0
                    elemC = 0
                    elemD = 0

                    if TE:
                        elemTE = np.sum(P02*clthn[3])
                        elemA += cji*elemTE
                        elemB -= sji*elemTE
                        elemC += cij*elemTE
                        elemD -= sij*elemTE

                    if TB:
                        elemTB = np.sum(P02*clthn[5])
                        elemA += sji*elemTB
                        elemB += cji*elemTB
                        elemC += sij*elemTB
                        elemD += cij*elemTB

                    S[i, jj] = elemA
                    S[i, jj+npi] = elemB
                    S[ii, j] = elemC
                    S[ii+npi, j] = elemD

                    S[jj, i] = elemA
                    S[jj+npi, i] = elemB
                    S[j, ii] = elemC
                    S[j, ii+npi] = elemD

                elem1 = np.sum((cij*cji*Q22 + sij*sji*R22)*clthn[1])
                elem2 = np.sum((-cij*sji*Q22 + sij*cji*R22)*clthn[1])
                elem3 = np.sum((sij*sji*Q22 + cij*cji*R22)*clthn[1])
                elem4 = np.sum((-sij*cji*Q22 + cij*sji*R22)*clthn[1])

                elem3 += np.sum((cij*cji*Q22 + sij*sji*R22)*clthn[2])
                elem4 -= np.sum((-cij*sji*Q22 + sij*cji*R22)*clthn[2])
                elem1 += np.sum((sij*sji*Q22 + cij*cji*R22)*clthn[2])
                elem2 -= np.sum((-sij*cji*Q22 + cij*sji*R22)*clthn[2])

                if EB:
                    elemEB = np.sum((Q22 - R22)*clthn[4])
                    elem1 += (cji*sij + sji*cij)*elemEB
                    elem2 += (-sji*sij + cji*cij)*elemEB
                    elem3 += (-sji*cij - cji*sij)*elemEB
                    elem4 += (cji*cij - sji*sij)*elemEB

                S[ii, jj] = elem1
                S[ii, jj+npi] = elem2
                S[ii+npi, jj+npi] = elem3
                S[ii+npi, jj] = elem4

                S[jj, ii] = elem1
                S[jj+npi, ii] = elem2
                S[jj+npi, ii+npi] = elem3
                S[jj, ii+npi] = elem4

                for b in np.arange(nbins):
                    elem1 = np.sum(( cij*cji*Q22+sij*sji*R22)[masks[b]])
                    elem2 = np.sum((-cij*sji*Q22+sij*cji*R22)[masks[b]])
                    elem3 = np.sum(( sij*sji*Q22+cij*cji*R22)[masks[b]])
                    elem4 = np.sum((-sij*cji*Q22+cij*sji*R22)[masks[b]])

                    # # EE ij then ji
                    Pl[ponbins + b, ii,     jj    ] = elem1
                    Pl[ponbins + b, ii,     jj+npi] = elem2
                    Pl[ponbins + b, ii+npi, jj+npi] = elem3
                    Pl[ponbins + b, ii+npi, jj    ] = elem4

                    Pl[ponbins + b, jj,     ii    ] = elem1
                    Pl[ponbins + b, jj+npi, ii    ] = elem2
                    Pl[ponbins + b, jj+npi, ii+npi] = elem3
                    Pl[ponbins + b, jj,     ii+npi] = elem4

                    # # BB ij then ji
                    Pl[ponbins + nbins+b, ii+npi, jj+npi] =  elem1
                    Pl[ponbins + nbins+b, ii+npi, jj    ] = -elem2
                    Pl[ponbins + nbins+b, ii,     jj    ] =  elem3
                    Pl[ponbins + nbins+b, ii,     jj+npi] = -elem4

                    Pl[ponbins + nbins+b, jj+npi, ii+npi] =  elem1
                    Pl[ponbins + nbins+b, jj,     ii+npi] = -elem2
                    Pl[ponbins + nbins+b, jj,     ii    ] =  elem3
                    Pl[ponbins + nbins+b, jj+npi, ii    ] = -elem4

                    if TE or TB:
                        elemA = np.sum(cji*P02[masks[b]])
                        elemB = np.sum(sji*P02[masks[b]])
                        elemC = np.sum(cij*P02[masks[b]])
                        elemD = np.sum(sij*P02[masks[b]])

                        if TE:
                            Pl[tenbins + b, i, jj    ] =  elemA
                            Pl[tenbins + b, i, jj+npi] = -elemB
                            Pl[tenbins + b, ii,     j] =  elemC
                            Pl[tenbins + b, ii+npi, j] = -elemD

                            Pl[tenbins + b, jj,     i] =  elemA
                            Pl[tenbins + b, jj+npi, i] = -elemB
                            Pl[tenbins + b, j,     ii] =  elemC
                            Pl[tenbins + b, j, ii+npi] = -elemD

                        if TB:
                            Pl[tbnbins + b, i, jj    ] = elemB
                            Pl[tbnbins + b, i, jj+npi] = elemA
                            Pl[tbnbins + b, ii, j    ] = elemD
                            Pl[tbnbins + b, ii+npi, j] = elemC

                            Pl[tbnbins + b, jj,     i] = elemB
                            Pl[tbnbins + b, jj+npi, i] = elemA
                            Pl[tbnbins + b, j,     ii] = elemD
                            Pl[tbnbins + b, j, ii+npi] = elemC

                    if EB:
                        Pl[ebnbins+b, ii,         jj] = -elem2-elem4
                        Pl[ebnbins+b, ii,     jj+npi] =  elem1-elem3
                        Pl[ebnbins+b, ii+npi, jj+npi] =  elem2+elem4
                        Pl[ebnbins+b, ii+npi,     jj] =  elem1-elem3

                        Pl[ebnbins+b, jj,         ii] = -elem2-elem4
                        Pl[ebnbins+b, jj+npi,     ii] =  elem1-elem3
                        Pl[ebnbins+b, jj+npi, ii+npi] =  elem2+elem4
                        Pl[ebnbins+b, jj,     ii+npi] =  elem1-elem3

    return Pl, S


def compute_S(
        ellbins, nside, ipok, allcosang, bl, clth, Slmax,
        spec, pixwin=True, timing=False):
    """
    Computes signal matrix S.

    Parameters
    ----------
    ellbins : array of floats
        Lowers bounds of bins. Example : taking ellbins = (2, 3, 5) will
        compute the spectra for bins = (2, 3.5).
    nside : int
        Healpix map resolution
    ipok : array of ints
        Healpy pixels numbers considered
    bl : 1D array of floats
        Beam window function
    clth : 4D or 6D array of float
        Fiducial power spectra
    Slmax : int
        Maximum lmax computed for the pixel covariance pixel matrix
    spec : 1D array of string
        Spectra list
    pixwin : bool
        If True, multiplies the beam window function by the pixel
        window function. Default: True
    timing : bool
        If True, displays timmer. Default: False

    Returns
    ----------
    S :  2D array of floats
        Pixel signal covariance matrix S

    Example
    ----------
    >>> S = compute_S(np.array([2,3,4,10]),4,ipok=np.array([0,1,3]),
    ... allcosang=np.linspace(0,1,15).reshape(3,-1), bl=np.arange(10),
    ... clth=np.arange(6*13).reshape(6,-1), Slmax=9,
    ... spec=["TT", "EE", "BB", "TE", "EB", "TB"], pixwin=True, timing=False)
    >>> print(round(np.sum(S),5))
    -17502.8982

    >>> S = compute_S(np.array([2,3,4,10]),4,ipok=np.array([0,1,3]),
    ... allcosang=np.linspace(0,1,15).reshape(3,-1), bl=np.arange(13),
    ... clth=np.arange(6*13).reshape(6,-1), Slmax=9,
    ... spec=["TT", "EE", "BB", "TE", "EB", "TB"], pixwin=False, timing=False)
    >>> print(round(np.sum(S),5))
    -31333.69722
    """

    lmax = ellbins[-1]
    ell = np.arange(np.min(ellbins), np.max(ellbins) + 1)
    nbins = len(ellbins) - 1
    minell = np.array(ellbins[0:nbins])
    maxell = np.array(ellbins[1:nbins+1]) - 1
    ellval = (minell + maxell) * 0.5

    npi = ipok.size
    stokes, spec, istokes, ispecs = getstokes(spec)
    nspec = len(spec)
    nsto = len(stokes)
    temp = "TT" in spec
    polar = "EE" in spec or "BB" in spec
    TE = 'TE' in spec
    EB = 'EB' in spec
    TB = 'TB' in spec
    te = spec.index('TE') if TE else 0
    tb = spec.index('TB') if TB else 0
    eb = spec.index('EB') if EB else 0
    ponbins = nbins*temp
    ponpi = npi*temp
    tenbins = te*nbins
    tbnbins = tb*nbins
    ebnbins = eb*nbins

    rpix = np.array(hp.pix2vec(nside, ipok))
    ll = np.arange(Slmax+1)
    fpixwin = extrapolpixwin(nside, Slmax, pixwin)
    norm = (2*ll[2:]+1)/(4.*np.pi)*(fpixwin[2:Slmax+1]**2)*(bl[2:Slmax+1]**2)
    clthn = clth[:, 2: Slmax+1]
    S = np.zeros((nsto*npi, nsto*npi))

    start = timeit.default_timer()
    for i in np.arange(npi):
        if timing:
            progress_bar(i, npi, -0.25 * (start-timeit.default_timer()))
        for j in np.arange(i, npi):
            if temp:
                pl = norm*pl0(allcosang[i, j], Slmax)[2:]
                elem = np.sum((pl * clthn[0]))
                S[i, j] = elem
                S[j, i] = elem

            if polar:
                ii = i+ponpi
                jj = j+ponpi
                cij, sij = polrotangle(rpix[:, i], rpix[:, j])
                cji, sji = polrotangle(rpix[:, j], rpix[:, i])
                cos_chi = allcosang[i, j]

                # Tegmark version
                Q22 = norm*F1l2(cos_chi, Slmax)
                # # /!\ signe - !
                R22 = -norm*F2l2(cos_chi, Slmax)

                # # Matt version
                # d20  = dlss(cos_chi, 2,  0, Slmax+1)
                # d2p2 = dlss(cos_chi, 2,  2, Slmax+1)
                # d2m2 = dlss(cos_chi, 2, -2, Slmax+1)
                # P02 = -d20[2:]
                # Q22 = ( d2p2 + d2m2 )[2:]/2.
                # R22 = ( d2p2 - d2m2 )[2:]/2.

                if TE or TB:
                    P02 = -norm*F1l0(cos_chi, Slmax)
                    elemA = 0
                    elemB = 0
                    elemC = 0
                    elemD = 0

                    if TE:
                        elemTE = P02*clthn[3]
                        elemA += np.sum(cji*elemTE)
                        elemB -= np.sum(sji*elemTE)
                        elemC += np.sum(cij*elemTE)
                        elemD -= np.sum(sij*elemTE)

                    if TB:
                        elemTB = P02*clthn[5]
                        elemA += np.sum(sji*elemTB)
                        elemB += np.sum(cji*elemTB)
                        elemC += np.sum(sij*elemTB)
                        elemD += np.sum(cij*elemTB)

                    S[i, jj] = elemA
                    S[i, jj+npi] = elemB
                    S[ii, j] = elemC
                    S[ii+npi, j] = elemD

                    S[jj, i] = elemA
                    S[jj+npi, i] = elemB
                    S[j, ii] = elemC
                    S[j, ii+npi] = elemD

                elem1 = np.sum((cij*cji*Q22 + sij*sji*R22)*clthn[1])
                elem2 = np.sum((-cij*sji*Q22 + sij*cji*R22)*clthn[1])
                elem3 = np.sum((sij*sji*Q22 + cij*cji*R22)*clthn[1])
                elem4 = np.sum((-sij*cji*Q22 + cij*sji*R22)*clthn[1])

                elem3 += np.sum((cij*cji*Q22 + sij*sji*R22)*clthn[2])
                elem4 -= np.sum((-cij*sji*Q22 + sij*cji*R22)*clthn[2])
                elem1 += np.sum((sij*sji*Q22 + cij*cji*R22)*clthn[2])
                elem2 -= np.sum((-sij*cji*Q22 + cij*sji*R22)*clthn[2])

                if EB:
                    elemEB = np.sum((Q22 - R22)*clthn[4])
                    elem1 += (cji*sij + sji*cij)*elemEB
                    elem2 += (-sji*sij + cji*cij)*elemEB
                    elem3 += (-sji*cij - cji*sij)*elemEB
                    elem4 += (cji*cij - sji*sij)*elemEB

                S[ii, jj] = elem1
                S[ii, jj+npi] = elem2
                S[ii+npi, jj+npi] = elem3
                S[ii+npi, jj] = elem4

                S[jj, ii] = elem1
                S[jj+npi, ii] = elem2
                S[jj+npi, ii+npi] = elem3
                S[jj, ii+npi] = elem4

    return S


def covth_bins_MC(
        ellbins, nside, ipok, allcosang, bl, clth, Slmax, nsimu,
        polar=False, temp=False, corr=False, pixwin=False, timing=False):
    """
    Can be particularly slow on sl7.
    To be enhanced and extended to TT and correlations.

    Parameters
    ----------
    ellbins : array of floats
        Lowers bounds of bins. Example : taking ellbins = (2, 3, 5) will
        compute the spectra for bins = (2, 3.5).
    nside : int
        Healpix map resolution
    ipok : array of ints
        Healpy pixels numbers considered
    bl : 1D array of floats
        Beam window function
    clth : 4D or 6D array of float
        Fiducial power spectra
    Slmax : int
        Maximum lmax computed for the pixel covariance pixel matrix
    spec : 1D array of string
        Spectra list
    pixwin : bool
        If True, multiplies the beam window function by the pixel
        window function. Default: True
    timing : bool
        If True, displays timmer. Default: False

    Returns
    ----------
    Pl : ndarray of floats
        Normalize Legendre polynomials dS/dCl

    S : 2D square matrix array of floats
        Pixel signal covariance matrix S

    Example
    ----------
    >>> import pylab
    >>> pylab.seed(0)
    >>> Pl, S = covth_bins_MC(np.array([2,3,4]), 4, ipok=np.array([0,1,3]),
    ... allcosang=np.linspace(0,1,13), bl=np.arange(13),
    ... clth=np.arange(4*13).reshape(4,-1), Slmax=11, nsimu=100,
    ... polar=True, temp=False, corr=False, pixwin=True, timing=False)
    >>> print(round(np.sum(Pl),5), round(np.sum(S),5))
    (12.68135, 406990.12056)

    >>> import pylab
    >>> pylab.seed(0)
    >>> Pl, S = covth_bins_MC(np.array([2,3,4]), 4, ipok=np.array([0,1,3]),
    ... allcosang=np.linspace(0,1,13), bl=np.arange(13),
    ... clth=np.arange(4*13).reshape(4,-1), Slmax=11, nsimu=100,
    ... polar=False, temp=True, corr=False, pixwin=True, timing=False)
    >>> print(round(np.sum(Pl),5), round(np.sum(S),5))
    (42.35592, 7414.01784)
    """
    if nsimu == 1:
        nsimu = (12 * nside**2) * 10 * (int(polar) + 1)

    lmax = ellbins[-1]
    ell = np.arange(np.min(ellbins), np.max(ellbins) + 1)
    nbins = len(ellbins) - 1
    minell = np.array(ellbins[0: nbins])
    maxell = np.array(ellbins[1: nbins + 1]) - 1
    ellval = (minell + maxell) * 0.5

    Stokes, spec, istokes, ispecs = getstokes(polar=polar, temp=temp, corr=corr)
    nspec = len(spec)
    ll = np.arange(Slmax+1)
    fpixwin = extrapolpixwin(nside, Slmax, pixwin)
    masks = []
    for i in np.arange(nbins):
        masks.append((ll[:] >= minell[i]) & (ll[:] <= maxell[i]))
    masks = np.array(masks)
    npix = len(ipok)
    start = timeit.default_timer()
    norm = bl[0: Slmax + 1]**2 * fpixwin[0: Slmax + 1]**2

    if polar:
        ClthOne = np.zeros((nspec * (nbins), 6, (Slmax + 1)))
        for l in np.arange(2 * nbins):
            ClthOne[l, int(l / nbins + 1)] = masks[l % nbins] * norm
        if corr:
            print("not implemented")
            # break;
            for l in np.arange(2 * nbins, 3 * nbins):
                ClthOne[l, 1] = masks[l % nbins] * norm
                ClthOne[l, 2] = masks[l % nbins] * norm
                ClthOne[l, 4] = masks[l % nbins] * norm

        Pl = np.zeros((nspec * (nbins), 2 * npix, 2 * npix))
        start = timeit.default_timer()
        for l in np.arange((nspec * nbins)):
            if timing:
                progress_bar(l, nspec * (nbins),
                             -(start - timeit.default_timer()))

            data = [
                np.array(
                    hp.synfast(
                        ClthOne[l], nside, lmax=Slmax, new=True, verbose=False)
                    )[1: 3, ipok].flatten() for s in np.arange(nsimu)]

            Pl[l] = np.cov(
                np.array(data).reshape(nsimu, 2 * npix), rowvar=False)

        Pl = Pl.reshape(nspec, nbins, 2 * npix, 2 * npix)
        S = np.cov(
            np.array([
                np.array(
                    hp.synfast(
                        clth[:, : Slmax + 1] * norm,
                        nside,
                        lmax=Slmax,
                        new=True,
                        verbose=False)
                    )[1:3, ipok].flatten() for s in np.arange(nsimu)]).reshape(
                        nsimu, 2*npix), rowvar=False)

    else:
        ClthOne = np.zeros((nbins, (Slmax + 1)))
        for l in np.arange((nbins)):
            ClthOne[l] = masks[l] * norm
        Pl = np.zeros(((nbins), npix, npix))
        for l in np.arange((nbins)):
            if timing:
                progress_bar(l, nspec * (nbins),
                             -(start - timeit.default_timer()))
            Pl[l] = np.cov(
                np.array([
                    hp.synfast(
                        ClthOne[l],
                        nside,
                        lmax=Slmax,
                        verbose=False
                        )[ipok] for s in np.arange(nsimu)]).reshape(
                            nsimu, npix), rowvar=False)

        Pl = Pl.reshape(1, nbins, npix, npix)
        S = np.cov(
            np.array([
                np.array(
                    hp.synfast(
                        clth[:, : Slmax + 1] * norm,
                        nside,
                        lmax=Slmax,
                        new=True,
                        verbose=False)
                    )[0, ipok].flatten() for s in np.arange(nsimu)]).reshape(
                        nsimu, npix), rowvar=False)

    stop = timeit.default_timer()
    return (Pl, S)


def S_bins_MC(
        ellbins, nside, ipok, allcosang, bl, clth, Slmax, nsimu,
        polar=True, temp=True, corr=False, pixwin=False, timing=False):
    """
    Can be particularly slow on sl7 !
    To be enhanced and extended to TT and correlations

    Parameters
    ----------
    ellbins : array of floats
        Lowers bounds of bins. Example : taking ellbins = (2, 3, 5) will
        compute the spectra for bins = (2, 3.5).
    nside : int
        Healpix map resolution
    ipok : array of ints
        Healpy pixels numbers considered
    bl : 1D array of floats
        Beam window function
    clth : 4D or 6D array of float
        Fiducial power spectra
    Slmax : int
        Maximum lmax computed for the pixel covariance pixel matrix
    polar : bool
        If True, get Stokes parameters for polar. Default: True
    temp : bool
        If True, get Stokes parameters for temperature. Default: False
    corr : bool
        If True, get Stokes parameters for EB and TB. Default: False
    pixwin : bool
        If True, multiplies the beam window function by the pixel
        window function. Default: True
    timing : bool
        If True, displays timmer. Default: False

    Returns
    ----------
    S :  2D array of floats
        Pixel signal covariance matrix S

    """
    if nsimu == 1:
        nsimu = (12 * nside**2) * 10 * (int(polar) + 1)
    lmax = ellbins[-1]
    ell = np.arange(np.min(ellbins), np.max(ellbins) + 1)
    nbins = len(ellbins) - 1
    minell = np.array(ellbins[0: nbins])
    maxell = np.array(ellbins[1: nbins + 1]) - 1
    ellval = (minell + maxell) * 0.5
    Stokes, spec, ind = getstokes(polar=polar, temp=temp, corr=corr)
    nspec = len(spec)
    ll = np.arange(Slmax + 2)
    fpixwin = extrapolpixwin(nside, Slmax, pixwin)
    masks = []
    for i in np.arange(nbins):
        masks.append((ll[:] >= minell[i]) & (ll[:] <= maxell[i]))
    masks = np.array(masks)
    npix = len(ipok)
    start = timeit.default_timer()
    norm = bl[0: Slmax + 1]**2 * fpixwin[0: Slmax + 1]**2

    if polar:
        ClthOne = np.zeros((nspec * (nbins), 6, (Slmax + 2)))
        for l in np.arange(2 * nbins):
            ClthOne[l, l / nbins + 1] = masks[l % nbins] * norm
        if corr:
            print("not implemented")
            # break;
            for l in np.arange(2*nbins, 3*nbins):
                ClthOne[l, 1] = masks[l % nbins]*norm
                ClthOne[l, 2] = masks[l % nbins]*norm
                # couille ici : -nbins*(l/nbins)]*norm
                ClthOne[l, 4] = masks[l % nbins]*norm

        S = np.cov(
            np.array(
                [np.array(hp.synfast(
                    clth[:, : Slmax + 2]*norm,
                    nside,
                    lmax=Slmax,
                    new=True,
                    verbose=False))[1:3, ipok].flatten()
                 for s in np.arange(nsimu)]).reshape(nsimu, 2 * npix),
            rowvar=False)

    else:
        ClthOne = np.zeros((nbins, (Slmax+2)))
        for l in np.arange((nbins)):
            ClthOne[l] = masks[l]*norm
        S = np.cov(
            np.array(
                [np.array(hp.synfast(
                    clth[:, : Slmax + 2] * norm,
                    nside,
                    lmax=Slmax,
                    new=True,
                    verbose=False))[0, ipok].flatten()
                 for s in np.arange(nsimu)]).reshape(nsimu, npix),
            rowvar=False)

    return S


def polrotangle(ri, rj):
    """
    Computes cosine and sine of twice the angle between pixels i and j.

    Parameters
    ----------
    ri : 3D array of floats
        Coordinates of vector corresponding to input pixels i following
        healpy.pix2vec(nside,ipix) output
    rj : 3D array of floats
        Coordinates of vector corresponding to input pixels j following
        healpy.pix2vec(nside,jpix) output

    Returns
    ----------
    cos2a : 1D array of floats
        Cosine of twice the angle between pixels i and j
    sin2a : 1D array of floats
        Sine of twice the angle between pixels i and j

    Example
    ----------
    >>> cos2a, sin2a = polrotangle([0.1,0.2,0.3], [0.4,0.5,0.6])
    >>> print(round(cos2a,5),round(sin2a,5))
    (0.06667, 0.37333)
    """
    z = np.array([0.0, 0.0, 1.0])

    # Compute ri^rj : unit vector for the great circle connecting i and j
    rij = np.cross(ri, rj)
    norm = np.sqrt(np.dot(rij, np.transpose(rij)))

    # case where pixels are identical or diametrically opposed on the sky
    if norm <= 1e-15:
        cos2a = 1.0
        sin2a = 0.0
        return cos2a, sin2a
    rij = rij / norm

    # Compute z^ri : unit vector for the meridian passing through pixel i
    ris = np.cross(z, ri)
    norm = np.sqrt(np.dot(ris, np.transpose(ris)))

    # case where pixels is at the pole
    if norm <= 1e-15:
        cos2a = 1.0
        sin2a = 0.0
        return cos2a, sin2a
    ris = ris / norm

    # Now, the angle we want is that
    # between these two great circles: defined by
    cosa = np.dot(rij, np.transpose(ris))

    # the sign is more subtle : see tegmark et de oliveira costa 2000 eq. A6
    rijris = np.cross(rij, ris)
    sina = np.dot(rijris, np.transpose(ri))

    # so now we have directly cos2a and sin2a
    cos2a = 2.0 * cosa * cosa - 1.0
    sin2a = 2.0 * cosa * sina

    return cos2a, sin2a


def dlss(z, s1, s2, lmax):
    """
    Computes the reduced Wigner D-function d^l_ss'

    Parameters
    ----------
    z : float
        Cosine of the angle between two pixels
    s1 : int
        Spin number 1
    s2 : int
        Spin number 2
    lmax : int
        Maximum multipole

    Returns
    ----------
    d : 1D array of floats
        ???

    Example
    ----------
    >>> d = dlss(0.1, 2,  2, 5)
    >>> print(round(sum(d),5))
    0.24351
    """
    d = np.zeros((lmax + 1))
    if s1 < abs(s2):
        print("error spins, s1<|s2|")
        return

    # Conv: sign = -1 if (s1 + s2) and 1 else 1
    sign = (-1)**(s1 - s2)
    fs1 = math.factorial(2.0 * s1)
    fs1ps2 = math.factorial(1.0 * s1 + s2)
    fs1ms2 = math.factorial(1.0 * s1 - s2)
    num = (1.0 + z)**(0.5 * (s1 + s2)) * (1.0 - z)**(0.5 * (s1 - s2))

    # Initialise the recursion (l = s1 + 1)
    d[s1] = sign / 2.0**s1 * np.sqrt(fs1 / fs1ps2 / fs1ms2) * num

    l1 = s1 + 1.0
    rhoSSL1 = np.sqrt((l1 * l1 - s1 * s1) * (l1 * l1 - s2 * s2)) / l1
    d[s1+1] = (2 * s1 + 1.0)*(z - s2 / (s1 + 1.0)) * d[s1] / rhoSSL1

    # Build the recursion for l > s1 + 1
    for l in np.arange(s1 + 1, lmax, 1):
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
    Computes the associated Legendre function of the first kind of order 0
    Pn(z) from 0 to lmax (inclusive).

    Parameters
    ----------
    z : float
        Cosine of the angle between two pixels
    lmax : int
        Maximum multipole

    Returns
    ----------
    Pn : 1D array of floats
    Legendre function

    Example
    ----------
    >>> thepl0 = pl0(0.1, 5)
    >>> print(round(sum(thepl0),5))
    0.98427
    """
    Pn = special.lpn(lmax, z)[0]
    return Pn


def pl2(z, lmax):
    """
    Computes the associated Legendre function of the first kind of order 2
    from 0 to lmax (inclusive)

    Parameters
    ----------
    z : float
        Cosine of the angle between two pixels
    lmax : int
        Maximum multipole

    Returns
    ----------
    Pn2 : 1D array of floats
    Legendre function

    Example
    ----------
    >>> thepl2 = pl2(0.1, 5)
    >>> print(round(sum(thepl2),5))
    -7.49183
    """
    Pn2 = special.lpmn(2, lmax, z)[0][2]
    return Pn2


# ####### F1 and F2 functions from Tegmark & De Oliveira-Costa, 2000  #########
def F1l0(z, lmax):
    """
    Compute the F1l0 function from Tegmark & De Oliveira-Costa, 2000

    Parameters
    ----------
    z : float
        Cosine of the angle between two pixels
    lmax : int
        Maximum multipole

    Returns
    ----------
    bla : 1D array of float
    F1l0 function from Tegmark & De Oliveira-Costa, 2000

    Example
    ----------
    >>> theF1l0= F1l0(0.1, 5)
    >>> print(round(sum(theF1l0),5))
    0.20392
    """
    if abs(z) == 1.0:
        return(np.zeros(lmax - 1))
    else:
        ell = np.arange(2, lmax + 1)
        thepl = pl0(z, lmax)
        theplm1 = np.append(0, thepl[:-1])
        thepl = thepl[2:]
        theplm1 = theplm1[2:]
        a0 = 2.0 / np.sqrt((ell - 1) * ell * (ell + 1) * (ell + 2))
        a1 = ell * z * theplm1 / (1 - z**2)
        a2 = (ell / (1 - z**2) + ell * (ell - 1) / 2) * thepl
        bla = a0 * (a1 - a2)
        return bla


def F1l2(z, lmax):
    """
    Compute the F1l2 function from Tegmark & De Oliveira-Costa, 2000

    Parameters
    ----------
    z : float
        Cosine of the angle between two pixels
    lmax : int
        Maximum multipole

    Returns
    ----------
    bla : 1D array of float
    F1l2 function from Tegmark & De Oliveira-Costa, 2000

    Example
    ----------
    >>> theF1l2= F1l2(0.1, 5)
    >>> print(round(sum(theF1l2),5))
    0.58396
    """
    if z == 1.0:
        return np.ones(lmax - 1) * 0.5
    elif z == -1.0:
        ell = np.arange(lmax + 1)
        return 0.5 * (-1)**ell[2:]
    else:
        ell = np.arange(2, lmax + 1)
        thepl2 = pl2(z, lmax)
        theplm1_2 = np.append(0, thepl2[:-1])
        thepl2 = thepl2[2:]
        theplm1_2 = theplm1_2[2:]
        a0 = 2.0 / ((ell - 1) * ell * (ell + 1) * (ell + 2))
        a1 = (ell + 2) * z * theplm1_2 / (1 - z**2)
        a2 = ((ell - 4) / (1 - z**2) + ell * (ell - 1) / 2) * thepl2
        bla = a0 * (a1 - a2)
        return bla


def F2l2(z, lmax):
    """
    Compute the F2l2 function from Tegmark & De Oliveira-Costa, 2000

    Parameters
    ----------
    z : float
        Cosine of the angle between two pixels
    lmax : int
        Maximum multipole

    Returns
    ----------
    ???

    Example
    ----------
    >>> theF2l2= F2l2(0.1, 5)
    >>> print(round(sum(theF2l2),5))
    0.34045
    """
    if z == 1.0:
        return -0.5 * np.ones(lmax - 1)
    elif z == -1.0:
        ell = np.arange(lmax + 1)
        return  0.5 * (-1)**ell[2:]
    else:
        ell = np.arange(2, lmax + 1)
        thepl2 = pl2(z, lmax)
        theplm1_2 = np.append(0, thepl2[:-1])
        thepl2 = thepl2[2:]
        theplm1_2 = theplm1_2[2:]
        a0 = 4.0 / ((ell - 1) * ell * (ell + 1) * (ell + 2) * (1 - z**2))
        a1 = (ell + 2) * theplm1_2
        a2 = (ell - 1) * z * thepl2
        bla = a0 * (a1 - a2)
        return bla



if __name__ == "__main__":
    """
    Run the doctest using

    python libcov.py'

    If the tests are OK, the script should exit gracefuly, otherwise the
    failure(s) will be printed out.
    """
    import doctest
    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")
    doctest.testmod()
