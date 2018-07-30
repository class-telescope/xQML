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
from xqml_utils import progress_bar

def compute_ds_dcb(
        ellbins, nside, ipok, bl, clth, Slmax,
        polar=True, temp=True, EBTB=False,
        pixwining=False, timing=False, MC=0, Sonly=False):
    """
    ???

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
    print('dS/dCb Calulation:')
    print('Temp=' + str(temp))
    print('Polar=' + str(polar))
    print('EBTB=' + str(EBTB))
    print('pixwining=' + str(pixwining))

    if Slmax < ellbins[-1]:
        print("WARNING : Slmax < lmax")

    # for max binning value Slmax+=1
    allStoke, der, ind = getstokes(polar=polar, temp=temp, EBTB=EBTB)
    print('Stokes parameters :', allStoke)
    print('Derivatives w.r.t. :', der)

    nder = len(der)

    #### define pixels
    rpix = np.array(hp.pix2vec(nside, ipok))
    allcosang = np.dot(np.transpose(rpix), rpix)
    allcosang[allcosang > 1] = 1.0
    allcosang[allcosang < -1] = -1.0

    ### dimensions of the large matrix and loop for filling it
    nstokes = len(allStoke)
    nbins = len(ellbins) - 1
    npix = len(ipok)

    start = timeit.default_timer()
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

    stop = timeit.default_timer()

    print("Time of computing : " + str(stop - start))

    return (dcov, Smatrix)

def covth_bins_MC(
        ellbins, nside, ipok, allcosang, bl, clth, Slmax, nsimu,
        polar=True, temp=True, EBTB=False, pixwining=False, timing=False):
    """
    Can be particularly slow on sl7 !

    Parameters
    ----------
    ellbins : ???
        ???

    Returns
    ----------
    ???

    """
    if nsimu == 1:
        nsimu = (12 * nside**2) * 10 * (int(polar) + 1)
    print("nsimu=", nsimu)
    lmax = ellbins[-1]
    ell = np.arange(np.min(ellbins), np.max(ellbins) + 1)
    # assure que ell.max < lmax
    maskl = ell < (lmax + 1)
    ell = ell[maskl]
    nbins = len(ellbins) - 1
    # define min
    minell = np.array(ellbins[0: nbins])
    # and max of a bin
    maxell = np.array(ellbins[1: nbins + 1]) - 1
    ellval = (minell + maxell) * 0.5

    print('minell:', minell)
    print('maxell:', maxell)

    # define Stokes
    allStoke, der, ind = getstokes(polar=polar, temp=temp, EBTB=EBTB)
    nder = len(der)

    # define pixels
    rpix = np.array(hp.pix2vec(nside, ipok))
    # assure que ell.max < lmax
    maskl = ell < (lmax + 1)

    # define Pixel window function
    # +1 avant Slmax+=1
    ll = np.arange(Slmax+2)

    if pixwining:
        prepixwin = np.array(hp.pixwin(nside, pol=True))
        rng = np.arange(len(prepixwin[1, 2:]))
        logpix = np.log(prepixwin[1, 2:])
        sqrtpix = np.sqrt(prepixwin[1, 2:])
        poly = np.polyfit(rng, logpix, deg=3, w=sqrtpix)
        # -1 avant Slmax+=1
        y_int = np.polyval(poly, np.arange(Slmax))
        fpixwin = np.append([0, 0], np.exp(y_int))
    else:
        fpixwin = ll * 0 + 1

    print("shape fpixwin", np.shape(fpixwin))
    # +1 avant Slmax+=1
    print("shape bl", np.shape(bl[: Slmax + 2]))
    masks = []
    for i in np.arange(nbins):
        masks.append((ll[:] >= minell[i]) & (ll[:] <= maxell[i]))
    masks = np.array(masks)
    print('Bins mask shape :', np.shape(masks))
    print(masks * 1)

    npix = len(ipok)
    start = timeit.default_timer()
    # +1 avant Slmax+=1
    norm = bl[0: Slmax + 2]**2 * fpixwin[0: Slmax + 2]**2

    if polar:
        # stok = [1,2,4] if EBTB else [1,2]
        # npix=2*npix
        # avant Slmax+=1
        ClthOne = np.zeros((nder * (nbins), 6, (Slmax + 2)))
        for l in np.arange(2 * nbins):
            ClthOne[l, l / nbins + 1] = masks[l % nbins] * norm
        if EBTB:
            print("not implemented")
            # break;
            for l in np.arange(2 * nbins, 3 * nbins):
                ClthOne[l, 1] = masks[l % nbins] * norm
                ClthOne[l, 2] = masks[l % nbins] * norm
                # couille ici: -nbins*(l/nbins)]*norm
                ClthOne[l, 4] = masks[l % nbins] * norm

        dcov = np.zeros((nder * (nbins), 2 * npix, 2 * npix))
        start = timeit.default_timer()
        for l in np.arange((nder * nbins)):
            progress_bar(l, nder * (nbins), -(start - timeit.default_timer()))

            # Synthetise map
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
            progress_bar(l, (lmax + 1), -(start - timeit.default_timer()))
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

    print("time [sec] = ", round(stop - start, 2))

    return (dcov, S)

def S_bins_MC(
        ellbins, nside, ipok, allcosang, bl, clth, Slmax, nsimu,
        polar=True,temp=True,EBTB=False, pixwining=False, timing=False):
    """
    Can be particularly slow on sl7 !

    Parameters
    ----------
    ellbins : ???
        ???

    Returns
    ----------
    ???
    """
    if nsimu == 1:
        nsimu = (12 * nside**2) * 10 * (int(polar) + 1)
    print("nsimu=", nsimu)
    lmax = ellbins[-1]
    ell = np.arange(np.min(ellbins), np.max(ellbins) + 1)
    # assure que ell.max < lmax
    maskl = ell < (lmax + 1)
    ell = ell[maskl]
    nbins = len(ellbins) - 1
    # define min
    minell = np.array(ellbins[0: nbins])
    # and max of a bin
    maxell = np.array(ellbins[1: nbins + 1]) - 1
    ellval = (minell + maxell) * 0.5

    print('minell:', minell)
    print('maxell:', maxell)

    # define Stokes
    allStoke, der, ind = getstokes(polar=polar, temp=temp, EBTB=EBTB)
    nder = len(der)

    # define pixels
    rpix = np.array(hp.pix2vec(nside, ipok))
    # assure que ell.max < lmax
    maskl = ell < (lmax + 1)

    # define Pixel window function
    # +1 avant Slmax+=1
    ll = np.arange(Slmax + 2)

    if pixwining:
        prepixwin = np.array(hp.pixwin(nside, pol=True))

        rng = np.arange(len(prepixwin[1, 2:]))
        logpix = np.log(prepixwin[1, 2:])
        sqrtpix = np.sqrt(prepixwin[1, 2:])
        poly = np.polyfit(rng, logpix, deg=3, w=sqrtpix)
        # -1 avant Slmax+=1
        y_int = np.polyval(poly, np.arange(Slmax))
        fpixwin = np.append([0, 0], np.exp(y_int))
        # fpixwin = np.array(hp.pixwin(nside, pol=True))[int(polar)][2:]
    else:
        # ???
        fpixwin = ll * 0 + 1

    print("shape fpixwin", np.shape(fpixwin))
    # +1 avant Slmax+=1
    print("shape bl", np.shape(bl[: Slmax + 2]))
    masks = []
    for i in np.arange(nbins):
        masks.append((ll[:] >= minell[i]) & (ll[:] <= maxell[i]))
    masks = np.array(masks)
    print('Bins mask shape :', np.shape(masks))
    print(masks * 1)

    npix = len(ipok)
    start = timeit.default_timer()
    # +1 avant Slmax+=1
    norm = bl[0: Slmax + 2]**2 * fpixwin[0: Slmax + 2]**2

    if polar:
        # stok = [1,2,4] if EBTB else [1,2]
        # npix=2*npix
        # avant Slmax+=1
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

    stop = timeit.default_timer()

    print("time [sec] = ", round(stop - start, 2))

    return S

def covth_bins_fast(
        ellbins, nside, ipok, allcosang, bl, clth, Slmax,
        polar=True, temp=True, EBTB=False, pixwining=False, timing=False):
    """
    Computes ds_dcb[nspec, nbins, 2*npix, 2*npix] and signal
    matrix S[2*npix, 2*npix].

    Fast because :
        - Building ds_dcb directly in the
          right shape [nspec, nbins, 2*npix, 2*npix]
        - Compute EE and BB parts of ds_dcb at
          the same time (using their symmetry properties).

    Parameters
    ----------
    ellbins : ???
        ???

    Returns
    ----------
    ???
    """

    #### define bins in ell
    lmax = ellbins[-1]
    ell = np.arange(np.min(ellbins), np.max(ellbins) + 1)
    # assure que ell.max < lmax
    maskl = ell < (lmax + 1)
    ell = ell[maskl]
    nbins = len(ellbins) - 1
    minell = np.array(ellbins[0:nbins])
    maxell = np.array(ellbins[1:nbins+1]) - 1
    ellval = (minell + maxell) * 0.5

    print('minell:', minell)
    print('maxell:', maxell)

    #### define Stokes
    allStoke, der, ind = getstokes(polar=polar, temp=temp, EBTB=EBTB)
    nder = len(der)

    #### define pixels
    rpix = np.array(hp.pix2vec(nside, ipok))

    # assure que ell.max < lmax
    maskl = ell < (lmax + 1)
    masklRQ = (np.arange(lmax+1) >= min(ell)) & (np.arange(lmax+1) < (lmax+1))

    # define Pixel window function
    ll = np.arange(Slmax+2)

    if pixwining:
        prepixwin = np.array(hp.pixwin(nside, pol=True))
        poly = np.polyfit(
            np.arange(
                len(prepixwin[int(polar), 2:])),
                np.log(prepixwin[int(polar), 2:]),
                deg=3,
                w=np.sqrt(prepixwin[int(polar), 2:]))
        y_int = np.polyval(poly, np.arange(Slmax))
        fpixwin = np.exp(y_int)
        fpixwin = np.append(
            prepixwin[int(polar)][2:],
            fpixwin[len(prepixwin[0]) - 2:])[: Slmax]
    else:
        fpixwin = ll[2:] * 0 + 1

    print("shape fpixwin", np.shape(fpixwin))
    print("shape bl", np.shape(bl[:Slmax+2]))
    # print(
    #     "long pixwin", fpixwin, "short",
    #     np.array(hp.pixwin(nside, pol=True))[int(polar)])
    norm = (2*ll[2:]+1)/(4.*np.pi)*(fpixwin**2)*(bl[2:Slmax+2]**2)
    print("norm ", np.shape(norm))
    print("ell ", np.shape(ell))

    #### define masks for ell bins
    masks = []
    for i in np.arange(nbins):
        masks.append((ll[2:] >= minell[i]) & (ll[2:] <= maxell[i]))
    masks = np.array(masks)
    print('Bins mask shape :', np.shape(masks))
    print("norm", norm)
    print("fpixwin", fpixwin)
    print("maskl", np.array(maskl)*1)
    print(masks*1)

    ### Create array for covariances matrices per bin
    nbpixok = ipok.size
    nstokes = np.size(allStoke)
    print('creating array')
    # final ds_dcb
    newcov = np.zeros((nder, nbins, nstokes*nbpixok, nstokes*nbpixok))
    # Signal matrix S
    Smatrix = np.zeros((nstokes*nbpixok, nstokes*nbpixok))

    start = timeit.default_timer()
    for i in np.arange(nbpixok):
        if timing:
            progress_bar(i, nbpixok, -0.5 * (start-timeit.default_timer()))
        for j in np.arange(i, nbpixok):
            if nstokes == 1:
                pl = pl0(allcosang[i, j], Slmax + 1)[2:]
                elem = np.sum((norm * pl * clth[0, 2: Slmax + 2])[:-1])
                Smatrix[i, j] = elem
                Smatrix[j, i] = elem
                for b in np.arange(nbins):
                    elem = np.sum((norm * pl)[masks[b]])
                    newcov[0, b, i, j] = elem
                    newcov[0, b, j, i] = elem

            elif nstokes == 2:
                cij, sij = polrotangle(rpix[:, i], rpix[:, j])
                cji, sji = polrotangle(rpix[:, j], rpix[:, i])
                cos_chi = allcosang[i, j]

                # Tegmark version
                # [masklRQ]
                Q22 = F1l2(cos_chi, Slmax + 1)[2:]
                # [masklRQ] # /!\ signe - !
                R22 = -F2l2(cos_chi, Slmax+1)[2:]

                # Matt version
                # d20  = dlss(cos_chi, 2,  0, Slmax+1)
                # d2p2 = dlss(cos_chi, 2,  2, Slmax+1)
                # d2m2 = dlss(cos_chi, 2, -2, Slmax+1)
                # P02 = -d20
                # Q22 = ( d2p2 + d2m2 )[2:]/2.
                # R22 = ( d2p2 - d2m2 )[2:]/2.

                # EE on QQ [masklRQ]
                elem1 = np.sum((norm * (
                    cij*cji*Q22 + sij*sji*R22)*(clth[1, 2: Slmax+2]))[: -1])
                # EE on QU [masklRQ]
                elem2 = np.sum((norm * (
                    -cij*sji*Q22 + sij*cji*R22)*(clth[1, 2: Slmax+2]))[: -1])
                # EE on UU [masklRQ]
                elem3 = np.sum((norm * (
                    sij*sji*Q22 + cij*cji*R22)*(clth[1, 2: Slmax+2]))[: -1])
                # EE on QU [masklRQ]
                elem4 = np.sum((norm * (
                    -sij*cji*Q22 + cij*sji*R22)*(clth[1, 2: Slmax+2]))[: -1])

                # BB on QQ [masklRQ]
                elem3 += np.sum((norm * (
                    cij*cji*Q22 + sij*sji*R22)*(clth[2, 2: Slmax+2]))[: -1])
                # BB on QU [masklRQ]
                elem4 -= np.sum((norm * (
                    -cij*sji*Q22 + sij*cji*R22)*(clth[2, 2: Slmax+2]))[: -1])
                # BB on UU [masklRQ]
                elem1 += np.sum((norm * (
                    sij*sji*Q22 + cij*cji*R22)*(clth[2, 2: Slmax+2]))[: -1])
                # BB on UQ [masklRQ]
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
                Smatrix[0*nbpixok+i, 0*nbpixok+j] = elem1
                # to -4
                Smatrix[0*nbpixok+i, 1*nbpixok+j] = elem2
                # to 1
                Smatrix[1*nbpixok+i, 1*nbpixok+j] = elem3
                # to -2
                Smatrix[1*nbpixok+i, 0*nbpixok+j] = elem4

                # to 3
                Smatrix[0*nbpixok+j, 0*nbpixok+i] = elem1
                # to -4
                Smatrix[1*nbpixok+j, 0*nbpixok+i] = elem2
                # to 1
                Smatrix[1*nbpixok+j, 1*nbpixok+i] = elem3
                # to -2
                Smatrix[0*nbpixok+j, 1*nbpixok+i] = elem4

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
                    newcov[0, b, i, j] = elem1
                    # to -4
                    newcov[0, b, i, nbpixok+j] = elem2
                    # to 1
                    newcov[0, b, nbpixok+i, nbpixok+j] = elem3
                    # to -2
                    newcov[0, b, nbpixok+i, j] = elem4

                    # to 3
                    newcov[0, b, j, i] = elem1
                    # to -4
                    newcov[0, b, nbpixok+j, i] = elem2
                    # to 1
                    newcov[0, b, nbpixok+j, nbpixok+i] = elem3
                    # to -2
                    newcov[0, b, j, nbpixok+i] = elem4

                    # # BB ij then ji
                    newcov[1, b, nbpixok+i, nbpixok+j] = elem1
                    newcov[1, b, nbpixok+i, j] = -elem2
                    newcov[1, b, i, j] = elem3
                    newcov[1, b, i, nbpixok+j] = -elem4

                    newcov[1, b, nbpixok+j, nbpixok+i] = elem1
                    newcov[1, b, j, nbpixok+i] = -elem2
                    newcov[1, b, j, i] = elem3
                    newcov[1, b, nbpixok+j, i] = -elem4

                    # # EB ij then ji
                    if EBTB:
                        # on QQ
                        newcov[2, b, i, j] = -elem2-elem4
                        # on QU
                        newcov[2, b, i, nbpixok+j] = elem1-elem3
                        # on UU
                        newcov[2, b, nbpixok+i, nbpixok+j] = elem2+elem4
                        # on UQ
                        newcov[2, b, nbpixok+i, j] = elem1-elem3

                        newcov[2, b, j, i] = -elem2-elem4
                        newcov[2, b, nbpixok+j, i] = elem1-elem3
                        newcov[2, b, nbpixok+j, nbpixok+i] = elem2+elem4
                        newcov[2, b, j, nbpixok+i] = elem1-elem3

                        # EB on all
                        elemQ22 = np.sum((norm * (Q22))[masks[b]])
                        # EB on all
                        elemR22 = np.sum((norm * (-R22))[masks[b]])
                        # on QQ
                        newcov[2, b, 0*nbpixok+i, 0*nbpixok+j] = (
                            sij*cji*(elemR22+elemQ22) +
                            cij*sji*(elemQ22+elemR22))
                        # on QU
                        newcov[2, b, 0*nbpixok+i, 1*nbpixok+j] = (
                            -sij*sji*(elemR22+elemQ22) +
                            cij*cji*(elemQ22+elemR22))
                        # on UU
                        newcov[2, b, 1*nbpixok+i, 1*nbpixok+j] = (
                            -cij*sji*(elemR22+elemQ22) -
                            sij*cji*(elemQ22+elemR22))
                        # on UQ
                        newcov[2, b, 1*nbpixok+i, 0*nbpixok+j] = (
                            cij*cji*(elemR22+elemQ22) -
                            sij*sji*(elemQ22+elemR22))
                        # to 3
                        newcov[2, b, 0*nbpixok+j, 0*nbpixok+i] = (
                            sij*cji*(elemR22+elemQ22) +
                            cij*sji*(elemQ22+elemR22))
                        # to -4
                        newcov[2, b, 1*nbpixok+j, 0*nbpixok+i] = (
                            -sij*sji*(elemR22+elemQ22) +
                            cij*cji*(elemQ22+elemR22))
                        # to 1
                        newcov[2, b, 1*nbpixok+j, 1*nbpixok+i] = (
                            -cij*sji*(elemR22+elemQ22) -
                            sij*cji*(elemQ22+elemR22))
                        # to -2
                        newcov[2, b, 0*nbpixok+j, 1*nbpixok+i] = (
                            cij*cji*(elemR22+elemQ22) -
                            sij*sji*(elemQ22+elemR22))

    return (newcov, Smatrix)

def S_bins_fast(
        ellbins, nside, ipok, allcosang, bl, clth, Slmax,
        polar=True,temp=True,EBTB=False, pixwining=False, timing=False):
    """
    Computes signal matrix S[2*npix, 2*npix]
    Fast because :
        - Compute EE and BB parts of ds_dcb at the same time
          (using their symmetry propreties).

    Parameters
    ----------
    ellbins : ???
        ???

    Returns
    ----------
    ???
    """

    lmax = ellbins[-1]
    ell = np.arange(np.min(ellbins), np.max(ellbins) + 1)
    # assure que ell.max < lmax
    maskl = ell < (lmax + 1)
    ell = ell[maskl]
    nbins = len(ellbins) - 1
    # define min
    minell = np.array(ellbins[0: nbins])
    maxell = np.array(ellbins[1: nbins + 1]) - 1
    ellval = (minell + maxell) * 0.5

    print('minell:', minell)
    print('maxell:', maxell)

    #### define Stokes
    allStoke = ['I', 'Q', 'U']
    if EBTB:
        der = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
        ind = [1, 2, 3, 4, 5, 6]
    else:
        der = ['TT', 'EE', 'BB', 'TE']
        ind = [1, 2, 3, 4]
    if not temp:
        allStoke = ['Q', 'U']
        if EBTB:
            der = ['EE', 'BB', 'EB']
            ind = [2, 3, 5]
        else:
            der = ['EE', 'BB']
            ind = [2, 3]
    if not polar:
        allStoke = ['I']
        der = ['TT']
        ind = [1]

    nder = len(der)

    # define pixels
    rpix = np.array(hp.pix2vec(nside, ipok))

    # assure que ell.max < lmax
    maskl = ell < (lmax + 1)
    masklRQ = (
        np.arange(lmax + 1) >= min(ell)) & (np.arange(lmax + 1) < (lmax + 1))

    # define Pixel window function
    ll = np.arange(Slmax + 2)

    if pixwining:
        prepixwin = np.array(hp.pixwin(nside, pol=True))
        poly = np.polyfit(
            np.arange(len(prepixwin[int(polar), 2:])),
            np.log(prepixwin[int(polar), 2:]),
            deg=3,
            w=np.sqrt(prepixwin[int(polar), 2:]))
        y_int = np.polyval(poly, np.arange(Slmax))
        fpixwin = np.exp(y_int)
        fpixwin = np.append(
            prepixwin[int(polar)][2:],
            fpixwin[len(prepixwin[0]) - 2:])[: Slmax]
    else:
        # ???
        fpixwin = ll[2:] * 0 + 1

    print("shape fpixwin", np.shape(fpixwin))
    print("shape bl", np.shape(bl[:Slmax+2]))
    norm = (2 * ll[2:]+1) / (4.*np.pi)*(fpixwin**2)*(bl[2: Slmax+2]**2)
    print("norm ", np.shape(norm))
    print("ell ", np.shape(ell))

    #### define masks for ell bins
    masks = []
    for i in np.arange(nbins):
        # masks.append((ell>=minell[i]) & (ell<=maxell[i]))
        masks.append((ll[2:] >= minell[i]) & (ll[2:] <= maxell[i]))
    masks = np.array(masks)
    print('Bins mask shape :', np.shape(masks))
    print("norm", norm)
    print("fpixwin", fpixwin)
    print("maskl", np.array(maskl)*1)
    print(masks*1)

    ### Create array for covariances matrices per bin
    nbpixok = ipok.size
    nstokes = np.size(allStoke)
    print('creating array')
    # Signal matrix S
    Smatrix = np.zeros((nstokes*nbpixok, nstokes*nbpixok))

    start = timeit.default_timer()
    for i in np.arange(nbpixok):
        if timing:
            progress_bar(i, nbpixok, -0.5*(start-timeit.default_timer()))
        for j in np.arange(i, nbpixok):
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
                # Q22 =  F1l2(cos_chi,Slmax+1)[2:] #[masklRQ]
                # R22 = -F2l2(cos_chi,Slmax+1)[2:] #[masklRQ] # /!\ signe - !

                # # # Matt version
                # d20  = dlss(cos_chi, 2,  0, Slmax+1)
                d2p2 = dlss(cos_chi, 2,  2, Slmax+1)
                d2m2 = dlss(cos_chi, 2, -2, Slmax+1)
                # # # P02 = -d20
                Q22 = (d2p2 + d2m2)[2:] / 2.0
                R22 = (d2p2 - d2m2)[2:] / 2.0

                # EE on QQ [masklRQ]
                elem1 = np.sum((norm * (
                    cij*cji*Q22 + sij*sji*R22)*(clth[1, 2: Slmax+2]))[: -1])
                # EE on QU [masklRQ]
                elem2 = np.sum((norm * (
                    -cij*sji*Q22 + sij*cji*R22)*(clth[1, 2: Slmax+2]))[:-1])
                # EE on UU [masklRQ]
                elem3 = np.sum((norm * (
                    sij*sji*Q22 + cij*cji*R22)*(clth[1, 2: Slmax+2]))[:-1])
                # EE on QU [masklRQ]
                elem4 = np.sum((norm * (
                    -sij*cji*Q22 + cij*sji*R22)*(clth[1, 2: Slmax+2]))[:-1])

                # BB on QQ [masklRQ]
                elem3 += np.sum((norm * (
                    cij*cji*Q22 + sij*sji*R22)*(clth[2, 2: Slmax+2]))[:-1])
                # BB on QU [masklRQ]
                elem4 -= np.sum((norm * (
                    -cij*sji*Q22 + sij*cji*R22)*(clth[2, 2: Slmax+2]))[:-1])
                # BB on UU [masklRQ]
                elem1 += np.sum((norm * (
                    sij*sji*Q22 + cij*cji*R22)*(clth[2, 2: Slmax+2]))[:-1])
                # BB on UQ [masklRQ]
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
                Smatrix[0*nbpixok+i, 0*nbpixok+j] = elem1
                # to -4
                Smatrix[0*nbpixok+i, 1*nbpixok+j] = elem2
                # to 1
                Smatrix[1*nbpixok+i, 1*nbpixok+j] = elem3
                # to -2
                Smatrix[1*nbpixok+i, 0*nbpixok+j] = elem4

                # to 3
                Smatrix[0*nbpixok+j, 0*nbpixok+i] = elem1
                # to -4
                Smatrix[1*nbpixok+j, 0*nbpixok+i] = elem2
                # to 1
                Smatrix[1*nbpixok+j, 1*nbpixok+i] = elem3
                # to -2
                Smatrix[0*nbpixok+j, 1*nbpixok+i] = elem4

    return (Smatrix)
