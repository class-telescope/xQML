"""
Set of routines to ...

Author: Vanneste
"""
from __future__ import division

import timeit

import numpy as np
import healpy as hp

from spin_functions import dlss, pl0
from spin_functions import F1l2, F2l2
from libangles import polrotangle
from simulation import getstokes
from simulation import extrapolpixwin
from xqml_utils import progress_bar


def compute_ds_dcb(
        ellbins, nside, ipok, bl, clth, Slmax,
        polar=True, temp=True, EBTB=False,
        pixwining=False, timing=False, MC=0, Sonly=False):
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
    polar : bool
        If True, get Stokes parameters for polar. Default: True
    temp : bool
        If True, get Stokes parameters for temperature. Default: False
    EBTB : bool
        If True, get Stokes parameters for EB and TB. Default: False
    pixwining : bool
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
    dcov : ndarray of floats
        Normalize Legendre polynomials dS/dCl

    Smatrix :  2D array of floats
        Pixel signal covariance matrix S

    Example
    ----------
    >>> Pl, S = compute_ds_dcb(
    ... np.array([2,4,5,8]), 2, np.array([0,1,4,10,11]), np.arange(10),
    ... clth=np.arange(30).reshape(3,-1), Slmax=8,
    ... polar=True, temp=False, EBTB=False,
    ... pixwining=False, timing=False, MC=0, Sonly=False)
    >>> print(round(np.sum(Pl),5), round(np.sum(S),5))
    (1158.35868, 38277.82933)

    >>> Pl, S = compute_ds_dcb(
    ... np.array([2,4,7,8]), 2, np.array([0,3,4,10,11]), np.arange(10),
    ... clth=np.arange(60).reshape(6,-1), Slmax=8,
    ... polar=True, temp=False, EBTB=True,
    ... pixwining=True, timing=False, MC=0, Sonly=False)
    >>> print(round(np.sum(Pl),5), round(np.sum(S),5))
    (570.84071, 15859.29629)

    >>> Pl, S = compute_ds_dcb(
    ... np.array([2,4,7,8]), 2, np.array([0,3,4,10,11]), np.arange(10),
    ... clth=np.arange(60).reshape(6,-1), Slmax=8,
    ... polar=False, temp=True, EBTB=False,
    ... pixwining=True, timing=False, MC=0, Sonly=False)
    >>> print(round(np.sum(Pl),5), round(np.sum(S),5))
    (109.73253, 679.79218)

    >>> import pylab
    >>> pylab.seed(0)
    >>> Pl, S = compute_ds_dcb(
    ... np.array([2,3,4]), 4, np.array([0,1,3]), np.arange(13),
    ... clth=np.arange(4*13).reshape(4,-1), Slmax=11,
    ... polar=True, temp=False, EBTB=False,
    ... pixwining=True, timing=False, MC=100, Sonly=False)
    >>> print(round(np.sum(Pl),5), round(np.sum(S),5))
    (12.68135, 406990.12056)

    """
    if Slmax < ellbins[-1]-1:
        print("WARNING : Slmax < lmax")

    allStoke, der, ind = getstokes(polar=polar, temp=temp, EBTB=EBTB)
    nder = len(der)

    # ### define pixels
    rpix = np.array(hp.pix2vec(nside, ipok))
    allcosang = np.dot(np.transpose(rpix), rpix)
    allcosang[allcosang > 1] = 1.0
    allcosang[allcosang < -1] = -1.0

    if Sonly:
        if MC:
            Smatrix = S_bins_MC(
                ellbins, nside, ipok, allcosang, bl, clth, Slmax, MC,
                polar=polar, temp=temp, EBTB=EBTB,
                pixwining=pixwining, timing=timing)
        else:
            Smatrix = S_bins_fast(
                ellbins, nside, ipok, allcosang, bl, clth, Slmax,
                polar=polar, temp=temp, EBTB=EBTB,
                pixwining=pixwining, timing=timing)
        return Smatrix

    if MC:
        dcov, Smatrix = covth_bins_MC(
            ellbins, nside, ipok, allcosang, bl, clth, Slmax, MC,
            polar=polar, temp=temp, EBTB=EBTB,
            pixwining=pixwining, timing=timing)
    else:
        dcov, Smatrix = covth_bins_fast(
            ellbins, nside, ipok, allcosang, bl, clth, Slmax,
            polar=polar, temp=temp, EBTB=EBTB,
            pixwining=pixwining, timing=timing)

    return (dcov, Smatrix)


def covth_bins_fast(
        ellbins, nside, ipok, allcosang, bl, clth, Slmax,
        polar=True, temp=True, EBTB=False, pixwining=False, timing=False):
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
    polar : bool
        If True, get Stokes parameters for polar. Default: True
    temp : bool
        If True, get Stokes parameters for temperature. Default: False
    EBTB : bool
        If True, get Stokes parameters for EB and TB. Default: False
    pixwining : bool
        If True, multiplies the beam window function by the pixel
        window function. Default: True
    timing : bool
        If True, displays timmer. Default: False

    Returns
    ----------
    dcov : ndarray of floats
        Normalize Legendre polynomials dS/dCl

    Smatrix :  2D array of floats
        Pixel signal covariance matrix S

    Example
    ----------
    >>> dcov, S = covth_bins_fast(np.array([2,3,4,7]),4,ipok=np.array([0,1,3]),
    ... allcosang=np.linspace(0,1,15).reshape(3,-1), bl=np.arange(13),
    ... clth=np.arange(4*13).reshape(4,-1), Slmax=11,
    ... polar=True, temp=False, EBTB=False, pixwining=True, timing=False)
    >>> print(round(np.sum(dcov),5), round(np.sum(S),5))
    (-22.04764, -2340.05934)


    >>> dcov, S = covth_bins_fast(np.array([2,3,4,7]),4,ipok=np.array([0,1,3]),
    ... allcosang=np.linspace(0,1,15).reshape(3,-1), bl=np.arange(13),
    ... clth=np.arange(6*13).reshape(6,-1), Slmax=11,
    ... polar=True, temp=False, EBTB=True, pixwining=True, timing=False)
    >>> print(round(np.sum(dcov),5), round(np.sum(S),5))
    (56.1748, 8597.38767)


    >>> dcov, S = covth_bins_fast(np.array([2,3,4,7]),4,ipok=np.array([0,1,3]),
    ... allcosang=np.linspace(0,1,15).reshape(3,-1), bl=np.arange(13),
    ... clth=np.arange(4*13).reshape(4,-1), Slmax=11,
    ... polar=False, temp=True, EBTB=False, pixwining=True, timing=False)
    >>> print(round(np.sum(dcov),5), round(np.sum(S),5))
    (-4.98779, -1270.80223)
    """

    lmax = ellbins[-1]
    ell = np.arange(np.min(ellbins), np.max(ellbins) + 1)
    nbins = len(ellbins) - 1
    minell = np.array(ellbins[0:nbins])
    maxell = np.array(ellbins[1:nbins+1]) - 1
    ellval = (minell + maxell) * 0.5

    allStoke, der, ind = getstokes(polar=polar, temp=temp, EBTB=EBTB)
    nder = len(der)
    rpix = np.array(hp.pix2vec(nside, ipok))
    ll = np.arange(Slmax+2)
    fpixwin = extrapolpixwin(nside, Slmax+2, pixwining)
    norm = (2*ll[2:]+1)/(4.*np.pi)*(fpixwin[2:]**2)*(bl[2:Slmax+2]**2)

    # ### define masks for ell bin
    masks = []
    for i in np.arange(nbins):
        masks.append((ll[2:] >= minell[i]) & (ll[2:] <= maxell[i]))
    masks = np.array(masks)

    # ## Create array for covariances matrices per bin
    hm = ipok.size
    nstokes = np.size(allStoke)
    newcov = np.zeros((nder*nbins, nstokes*hm, nstokes*hm))
    Smatrix = np.zeros((nstokes*hm, nstokes*hm))

    start = timeit.default_timer()
    for i in np.arange(hm):
        if timing:
            progress_bar(i, hm, -0.5 * (start-timeit.default_timer()))
        for j in np.arange(i, hm):
            if temp:
                pl = pl0(allcosang[i, j], Slmax + 1)[2:]
                elem = np.sum((norm * pl * clth[0, 2: Slmax + 2])[:-1])
                Smatrix[i, j] = elem
                Smatrix[j, i] = elem
                for b in np.arange(nbins):
                    elem = np.sum((norm * pl)[masks[b]])
                    newcov[b, i, j] = elem
                    newcov[b, j, i] = elem

            elif polar:
                cij, sij = polrotangle(rpix[:, i], rpix[:, j])
                cji, sji = polrotangle(rpix[:, j], rpix[:, i])
                cos_chi = allcosang[i, j]

                # Tegmark version
                Q22 = F1l2(cos_chi, Slmax + 1)[2:]
                # # /!\ signe - !
                R22 = -F2l2(cos_chi, Slmax+1)[2:]

                # Matt version
                # d20  = dlss(cos_chi, 2,  0, Slmax+1)
                # d2p2 = dlss(cos_chi, 2,  2, Slmax+1)
                # d2m2 = dlss(cos_chi, 2, -2, Slmax+1)
                # P02 = -d20
                # Q22 = ( d2p2 + d2m2 )[2:]/2.
                # R22 = ( d2p2 - d2m2 )[2:]/2.

                # EE on QQ
                elem1 = np.sum((norm * (
                    cij*cji*Q22 + sij*sji*R22)*(clth[1, 2: Slmax+2]))[: -1])
                # EE on QU
                elem2 = np.sum((norm * (
                    -cij*sji*Q22 + sij*cji*R22)*(clth[1, 2: Slmax+2]))[: -1])
                # EE on UU
                elem3 = np.sum((norm * (
                    sij*sji*Q22 + cij*cji*R22)*(clth[1, 2: Slmax+2]))[: -1])
                # EE on QU
                elem4 = np.sum((norm * (
                    -sij*cji*Q22 + cij*sji*R22)*(clth[1, 2: Slmax+2]))[: -1])

                # BB on QQ
                elem3 += np.sum((norm * (
                    cij*cji*Q22 + sij*sji*R22)*(clth[2, 2: Slmax+2]))[: -1])
                # BB on QU
                elem4 -= np.sum((norm * (
                    -cij*sji*Q22 + sij*cji*R22)*(clth[2, 2: Slmax+2]))[: -1])
                # BB on UU
                elem1 += np.sum((norm * (
                    sij*sji*Q22 + cij*cji*R22)*(clth[2, 2: Slmax+2]))[: -1])
                # BB on UQ
                elem2 -= np.sum((norm * (
                    -sij*cji*Q22 + cij*sji*R22)*(clth[2, 2: Slmax+2]))[: -1])

                if EBTB:
                    # EB on all
                    elem = np.sum(
                        (norm * (Q22 - R22)*(clth[4, 2: Slmax+2]))[: -1])
                    # EB on QQ
                    elem1 += (cji*sij + sji*cij)*elem
                    # EB on QU
                    elem2 += (-sji*sij + cji*cij)*elem
                    # EB on UU
                    elem3 += (-sji*cij - cji*sij)*elem
                    # EB on QU
                    elem4 += (cji*cij - sji*sij)*elem

                # to 3
                Smatrix[i, j] = elem1
                # to -4
                Smatrix[i, hm+j] = elem2
                # to 1
                Smatrix[hm+i, hm+j] = elem3
                # to -2
                Smatrix[hm+i, j] = elem4

                # to 3
                Smatrix[j, i] = elem1
                # to -4
                Smatrix[hm+j, i] = elem2
                # to 1
                Smatrix[hm+j, hm+i] = elem3
                # to -2
                Smatrix[j, hm+i] = elem4

                for b in np.arange(nbins):
                    # EE or BB on QQ
                    elem1 = np.sum((norm*(cij*cji*Q22+sij*sji*R22))[masks[b]])
                    # EE or BB on QU
                    elem2 = np.sum((norm*(-cij*sji*Q22+sij*cji*R22))[masks[b]])
                    # EE or BB on UU
                    elem3 = np.sum((norm*(sij*sji*Q22+cij*cji*R22))[masks[b]])
                    # EE or BB on UQ
                    elem4 = np.sum((norm*(-sij*cji*Q22+cij*sji*R22))[masks[b]])

                    # # EE ij then ji
                    # to 3 for BB
                    newcov[temp*nbins + b, i, j] = elem1
                    # to -4
                    newcov[temp*nbins + b, i, hm+j] = elem2
                    # to 1
                    newcov[temp*nbins + b, hm+i, hm+j] = elem3
                    # to -2
                    newcov[temp*nbins + b, hm+i, j] = elem4

                    # to 3
                    newcov[temp*nbins + b, j, i] = elem1
                    # to -4
                    newcov[temp*nbins + b, hm+j, i] = elem2
                    # to 1
                    newcov[temp*nbins + b, hm+j, hm+i] = elem3
                    # to -2
                    newcov[temp*nbins + b, j, hm+i] = elem4

                    # # BB ij then ji
                    newcov[temp*nbins + nbins+b, hm+i, hm+j] = elem1
                    newcov[temp*nbins + nbins+b, hm+i, j] = -elem2
                    newcov[temp*nbins + nbins+b, i, j] = elem3
                    newcov[temp*nbins + nbins+b, i, hm+j] = -elem4

                    newcov[temp*nbins + nbins+b, hm+j, hm+i] = elem1
                    newcov[temp*nbins + nbins+b, j, hm+i] = -elem2
                    newcov[temp*nbins + nbins+b, j, i] = elem3
                    newcov[temp*nbins + nbins+b, hm+j, i] = -elem4

                    # # EB ij then ji
                    if EBTB:
                        # on QQ
                        newcov[2*nbins+b, i, j] = -elem2-elem4
                        # on QU
                        newcov[2*nbins+b, i, hm+j] = elem1-elem3
                        # on UU
                        newcov[2*nbins+b, hm+i, hm+j] = elem2+elem4
                        # on UQ
                        newcov[2*nbins+b, hm+i, j] = elem1-elem3

                        newcov[2*nbins+b, j, i] = -elem2-elem4
                        newcov[2*nbins+b, hm+j, i] = elem1-elem3
                        newcov[2*nbins+b, hm+j, hm+i] = elem2+elem4
                        newcov[2*nbins+b, j, hm+i] = elem1-elem3

                        # EB on all
                        elemQ22 = np.sum((norm * (Q22))[masks[b]])
                        # EB on all
                        elemR22 = np.sum((norm * (-R22))[masks[b]])
                        # on QQ
                        newcov[2*nbins+b, i, j] = (
                            sij*cji*(elemR22+elemQ22) +
                            cij*sji*(elemQ22+elemR22))
                        # on QU
                        newcov[2*nbins+b, i, hm+j] = (
                            -sij*sji*(elemR22+elemQ22) +
                            cij*cji*(elemQ22+elemR22))
                        # on UU
                        newcov[2*nbins+b, hm+i, hm+j] = (
                            -cij*sji*(elemR22+elemQ22) -
                            sij*cji*(elemQ22+elemR22))
                        # on UQ
                        newcov[2*nbins+b, hm+i, j] = (
                            cij*cji*(elemR22+elemQ22) -
                            sij*sji*(elemQ22+elemR22))
                        # to 3
                        newcov[2*nbins+b, j, i] = (
                            sij*cji*(elemR22+elemQ22) +
                            cij*sji*(elemQ22+elemR22))
                        # to -4
                        newcov[2*nbins+b, hm+j, i] = (
                            -sij*sji*(elemR22+elemQ22) +
                            cij*cji*(elemQ22+elemR22))
                        # to 1
                        newcov[2*nbins+b, hm+j, hm+i] = (
                            -cij*sji*(elemR22+elemQ22) -
                            sij*cji*(elemQ22+elemR22))
                        # to -2
                        newcov[2*nbins+b, j, hm+i] = (
                            cij*cji*(elemR22+elemQ22) -
                            sij*sji*(elemQ22+elemR22))

    return (newcov, Smatrix)


def S_bins_fast(
        ellbins, nside, ipok, allcosang, bl, clth, Slmax,
        polar=True, temp=True, EBTB=False, pixwining=False, timing=False):
    """
    Computes pixel covariance signal matrix S.

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
    EBTB : bool
        If True, get Stokes parameters for EB and TB. Default: False
    pixwining : bool
        If True, multiplies the beam window function by the pixel
        window function. Default: True
    timing : bool
        If True, displays timmer. Default: False

    Returns
    ----------
    Smatrix : 2D array of floats
        Pixel signal covariance matrix S

    Example
    ----------
    >>> S = S_bins_fast(np.array([2,3,4,7]),4,ipok=np.array([0,1,3]),
    ... allcosang=np.linspace(0,1,15).reshape(3,-1), bl=np.arange(13),
    ... clth=np.arange(4*13).reshape(4,-1), Slmax=11,
    ... polar=True, temp=False, EBTB=False, pixwining=True, timing=False)
    >>> print(round(np.sum(S),5))
    -2340.05934

    >>> S = S_bins_fast(np.array([2,3,4,7]),4,ipok=np.array([0,1,3]),
    ... allcosang=np.linspace(0,1,15).reshape(3,-1), bl=np.arange(13),
    ... clth=np.arange(4*13).reshape(4,-1), Slmax=11,
    ... polar=False, temp=True, EBTB=False, pixwining=True, timing=False)
    >>> print(round(np.sum(S),5))
    -1270.80223
    """

    lmax = ellbins[-1]
    ell = np.arange(np.min(ellbins), np.max(ellbins) + 1)
    nbins = len(ellbins) - 1
    minell = np.array(ellbins[0: nbins])
    maxell = np.array(ellbins[1: nbins + 1]) - 1
    ellval = (minell + maxell) * 0.5
    allStoke, spec, ind = getstokes(polar=polar, temp=temp, EBTB=EBTB)
    nder = len(spec)

    # define pixels
    rpix = np.array(hp.pix2vec(nside, ipok))
    # define Pixel window function
    ll = np.arange(Slmax + 2)
    fpixwin = extrapolpixwin(nside, Slmax+2, pixwining)
    norm = (2*ll[2:]+1)/(4.*np.pi)*(fpixwin[2:]**2)*(bl[2:Slmax+2]**2)

    # ### define masks for ell bins
    masks = []
    for i in np.arange(nbins):
        masks.append((ll[2:] >= minell[i]) & (ll[2:] <= maxell[i]))
    masks = np.array(masks)

    # ## Create array for covariances matrices per bin
    hm = ipok.size
    nstokes = np.size(allStoke)
    Smatrix = np.zeros((nstokes*hm, nstokes*hm))
    for i in np.arange(hm):
        if timing:
            progress_bar(i, hm, -0.5*(start-timeit.default_timer()))
        for j in np.arange(i, hm):
            if nstokes == 1:
                pl = pl0(allcosang[i, j], Slmax+1)[2:]
                elem = np.sum((norm*pl*clth[0, 2: Slmax+2])[:-1])
                Smatrix[i, j] = elem
                Smatrix[j, i] = elem
            elif nstokes == 2:
                cij, sij = polrotangle(rpix[:, i], rpix[:, j])
                cji, sji = polrotangle(rpix[:, j], rpix[:, i])
                cos_chi = allcosang[i, j]

                # # # JC version
                # Q22 =  F1l2(cos_chi,Slmax+1)[2:] #
                # R22 = -F2l2(cos_chi,Slmax+1)[2:] # # /!\ signe - !

                # # # Matt version
                # d20  = dlss(cos_chi, 2,  0, Slmax+1)
                d2p2 = dlss(cos_chi, 2,  2, Slmax+1)
                d2m2 = dlss(cos_chi, 2, -2, Slmax+1)
                # # # P02 = -d20
                Q22 = (d2p2 + d2m2)[2:] / 2.0
                R22 = (d2p2 - d2m2)[2:] / 2.0

                # EE on QQ
                elem1 = np.sum((norm * (
                    cij*cji*Q22 + sij*sji*R22)*(clth[1, 2: Slmax+2]))[: -1])
                # EE on QU
                elem2 = np.sum((norm * (
                    -cij*sji*Q22 + sij*cji*R22)*(clth[1, 2: Slmax+2]))[:-1])
                # EE on UU
                elem3 = np.sum((norm * (
                    sij*sji*Q22 + cij*cji*R22)*(clth[1, 2: Slmax+2]))[:-1])
                # EE on QU
                elem4 = np.sum((norm * (
                    -sij*cji*Q22 + cij*sji*R22)*(clth[1, 2: Slmax+2]))[:-1])

                # BB on QQ
                elem3 += np.sum((norm * (
                    cij*cji*Q22 + sij*sji*R22)*(clth[2, 2: Slmax+2]))[:-1])
                # BB on QU
                elem4 -= np.sum((norm * (
                    -cij*sji*Q22 + sij*cji*R22)*(clth[2, 2: Slmax+2]))[:-1])
                # BB on UU
                elem1 += np.sum((norm * (
                    sij*sji*Q22 + cij*cji*R22)*(clth[2, 2: Slmax+2]))[:-1])
                # BB on UQ
                elem2 -= np.sum((norm * (
                    -sij*cji*Q22 + cij*sji*R22)*(clth[2, 2: Slmax+2]))[:-1])

                if EBTB:
                    # EB on all
                    elem = np.sum((
                        norm * (Q22 - R22)*(clth[4, 2: Slmax+2]))[:-1])
                    # EB on QQ
                    elem1 += (cji*sij + sji*cij)*elem
                    # EB on QU
                    elem2 += (-sji*sij + cji*cij)*elem
                    # EB on UU
                    elem3 += (-sji*cij - cji*sij)*elem
                    # EB on QU
                    elem4 += (cji*cij - sji*sij)*elem

                # to 3
                Smatrix[i, j] = elem1
                # to -4
                Smatrix[i, hm+j] = elem2
                # to 1
                Smatrix[hm+i, hm+j] = elem3
                # to -2
                Smatrix[hm+i, j] = elem4

                # to 3
                Smatrix[j, i] = elem1
                # to -4
                Smatrix[hm+j, i] = elem2
                # to 1
                Smatrix[hm+j, hm+i] = elem3
                # to -2
                Smatrix[j, hm+i] = elem4

    return (Smatrix)


def covth_bins_MC(
        ellbins, nside, ipok, allcosang, bl, clth, Slmax, nsimu,
        polar=True, temp=False, EBTB=False, pixwining=False, timing=False):
    """
    Can be particularly slow on sl7 !
    To be enhanced and extended to TT and EB

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
    EBTB : bool
        If True, get Stokes parameters for EB and TB. Default: False
    pixwining : bool
        If True, multiplies the beam window function by the pixel
        window function. Default: True
    timing : bool
        If True, displays timmer. Default: False

    Returns
    ----------
    dcov : ndarray of floats
        Normalize Legendre polynomials dS/dCl

    S : 2D square matrix array of floats
        Pixel signal covariance matrix S

    Example
    ----------
    >>> import pylab
    >>> pylab.seed(0)
    >>> dcov, S = covth_bins_MC(np.array([2,3,4]), 4, ipok=np.array([0,1,3]),
    ... allcosang=np.linspace(0,1,13), bl=np.arange(13),
    ... clth=np.arange(4*13).reshape(4,-1), Slmax=11, nsimu=100,
    ... polar=True, temp=False, EBTB=False, pixwining=True, timing=False)
    >>> print(round(np.sum(dcov),5), round(np.sum(S),5))
    (12.68135, 406990.12056)

    >>> import pylab
    >>> pylab.seed(0)
    >>> dcov, S = covth_bins_MC(np.array([2,3,4]), 4, ipok=np.array([0,1,3]),
    ... allcosang=np.linspace(0,1,13), bl=np.arange(13),
    ... clth=np.arange(4*13).reshape(4,-1), Slmax=11, nsimu=100,
    ... polar=False, temp=True, EBTB=False, pixwining=True, timing=False)
    >>> print(round(np.sum(dcov),5), round(np.sum(S),5))
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

    allStoke, der, ind = getstokes(polar=polar, temp=temp, EBTB=EBTB)
    nder = len(der)
    ll = np.arange(Slmax+2)
    fpixwin = extrapolpixwin(nside, Slmax+2, pixwining)
    masks = []
    for i in np.arange(nbins):
        masks.append((ll[:] >= minell[i]) & (ll[:] <= maxell[i]))
    masks = np.array(masks)
    npix = len(ipok)
    start = timeit.default_timer()
    norm = bl[0: Slmax + 2]**2 * fpixwin[0: Slmax + 2]**2

    if polar:
        ClthOne = np.zeros((nder * (nbins), 6, (Slmax + 2)))
        for l in np.arange(2 * nbins):
            ClthOne[l, int(l / nbins + 1)] = masks[l % nbins] * norm
        if EBTB:
            print("not implemented")
            # break;
            for l in np.arange(2 * nbins, 3 * nbins):
                ClthOne[l, 1] = masks[l % nbins] * norm
                ClthOne[l, 2] = masks[l % nbins] * norm
                ClthOne[l, 4] = masks[l % nbins] * norm

        dcov = np.zeros((nder * (nbins), 2 * npix, 2 * npix))
        start = timeit.default_timer()
        for l in np.arange((nder * nbins)):
            if timing:
                progress_bar(l, nder * (nbins),
                             -(start - timeit.default_timer()))

            data = [
                np.array(
                    hp.synfast(
                        ClthOne[l], nside, lmax=Slmax, new=True, verbose=False)
                    )[1: 3, ipok].flatten() for s in np.arange(nsimu)]

            dcov[l] = np.cov(
                np.array(data).reshape(nsimu, 2 * npix), rowvar=False)

        dcov = dcov.reshape(nder, nbins, 2 * npix, 2 * npix)
        S = np.cov(
            np.array([
                np.array(
                    hp.synfast(
                        clth[:, : Slmax + 2] * norm,
                        nside,
                        lmax=Slmax,
                        new=True,
                        verbose=False)
                    )[1:3, ipok].flatten() for s in np.arange(nsimu)]).reshape(
                        nsimu, 2*npix), rowvar=False)

    else:
        ClthOne = np.zeros((nbins, (Slmax + 2)))
        for l in np.arange((nbins)):
            ClthOne[l] = masks[l] * norm
        dcov = np.zeros(((nbins), npix, npix))
        for l in np.arange((nbins)):
            if timing:
                progress_bar(l, nder * (nbins),
                             -(start - timeit.default_timer()))
            dcov[l] = np.cov(
                np.array([
                    hp.synfast(
                        ClthOne[l],
                        nside,
                        lmax=Slmax,
                        verbose=False
                        )[ipok] for s in np.arange(nsimu)]).reshape(
                            nsimu, npix), rowvar=False)

        dcov = dcov.reshape(1, nbins, npix, npix)
        S = np.cov(
            np.array([
                np.array(
                    hp.synfast(
                        clth[:, : Slmax + 2] * norm,
                        nside,
                        lmax=Slmax,
                        new=True,
                        verbose=False)
                    )[0, ipok].flatten() for s in np.arange(nsimu)]).reshape(
                        nsimu, npix), rowvar=False)

    stop = timeit.default_timer()
    return (dcov, S)


def S_bins_MC(
        ellbins, nside, ipok, allcosang, bl, clth, Slmax, nsimu,
        polar=True, temp=True, EBTB=False, pixwining=False, timing=False):
    """
    Can be particularly slow on sl7 !
    In developpement to TT and EB

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
    EBTB : bool
        If True, get Stokes parameters for EB and TB. Default: False
    pixwining : bool
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
    allStoke, der, ind = getstokes(polar=polar, temp=temp, EBTB=EBTB)
    nder = len(der)
    ll = np.arange(Slmax + 2)
    fpixwin = extrapolpixwin(nside, Slmax+2, pixwining)
    masks = []
    for i in np.arange(nbins):
        masks.append((ll[:] >= minell[i]) & (ll[:] <= maxell[i]))
    masks = np.array(masks)
    npix = len(ipok)
    start = timeit.default_timer()
    norm = bl[0: Slmax + 2]**2 * fpixwin[0: Slmax + 2]**2

    if polar:
        ClthOne = np.zeros((nder * (nbins), 6, (Slmax + 2)))
        for l in np.arange(2 * nbins):
            ClthOne[l, l / nbins + 1] = masks[l % nbins] * norm
        if EBTB:
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

if __name__ == "__main__":
    """
    Run the doctest using

    python simulation.py'

    If the tests are OK, the script should exit gracefuly, otherwise the
    failure(s) will be printed out.
    """
    import doctest
    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")
    doctest.testmod()