"""
Test script for xQML

Author: Vanneste
"""
from __future__ import division

import timeit
import numpy as np
import healpy as hp
from pylab import *
import astropy.io.fits as fits

from xqml import xQML
from libcov import compute_ds_dcb
from xqml_utils import progress_bar
from simulation import getstokes, muKarcmin2var, GetBinningMatrix
from simulation import extrapolpixwin


if __name__ == "__main__":
    nside = 4
    lmax = 3 * nside - 1
    Slmax = 3 * nside - 1
    deltal = 1
    nsimu = 500
    clth = np.array(hp.read_cl('planck_base_planck_2015_TTlowP.fits'))
    lth = arange(2, lmax+1)

    ellbins = arange(2, lmax + 2, deltal)
    ellbins[-1] = lmax + 1

    P, Q, ell, ellval = GetBinningMatrix(ellbins, lmax)
    nbins = len(ellbins) - 1

    # Create mask
    t, p = hp.pix2ang(nside, range(hp.nside2npix(nside)))
    mask = np.ones(hp.nside2npix(nside), bool)
    mask[abs(90 - rad2deg(t)) < 10] = False
    npix = sum(mask)

    fwhm = 0.5
    bl = hp.gauss_beam(deg2rad(fwhm), lmax=Slmax + 1)

    allStoke, der, ind = getstokes(polar=True, temp=False, EBTB=False)
    nder = len(der)

    muKarcmin = 0.1

    pixvar = muKarcmin2var(muKarcmin, nside)
    varmap = ones((2 * npix)) * pixvar
    NoiseVar = np.diag(varmap)

    cmb = np.array(hp.synfast(
        clth, nside, fwhm=deg2rad(fwhm), pixwin=True, new=True, verbose=False))
    noise = (randn(len(varmap)) * varmap**0.5).reshape(2, -1)
    # dm = cmb[1:, mask] + noise
    dm = cmb[1:][:, mask] + noise

    # ############## Compute ds_dcb ###############
    ip = arange(hp.nside2npix(nside))
    ipok = ip[mask]

    Pl, S = compute_ds_dcb(
        ellbins, nside, ipok, bl, clth, Slmax,
        polar=True, temp=False, EBTB=False,
        pixwining=True, timing=True, MC=False)
    # Pl = Pl.reshape((nder)*(np.shape(Pl)[1]), 2 * npix, 2 * npix)

    # ############## Compute spectra ###############
    esti = xQML(mask, ellbins, clth, Pl=Pl, S=S, fwhm=fwhm)
    esti.construct_esti(NoiseVar, NoiseVar)
    cl = esti.get_spectra(dm, dm)
    V = esti.get_covariance()

    # ############## Construct MC ###############
    allcl = []
    esti = xQML(mask, ellbins, clth, Pl=Pl, fwhm=fwhm)
    esti.construct_esti(NoiseVar, NoiseVar)
    fpixw = extrapolpixwin(nside, lmax+2, pixwining=True)
    start = timeit.default_timer()
    for n in np.arange(nsimu):
        progress_bar(n, nsimu, timeit.default_timer() - start)
        cmb = np.array(hp.synfast(clth[:, :len(fpixw)]*fpixw**2, nside,
                       fwhm=deg2rad(fwhm), new=True, verbose=False))
        dm = cmb[1:, mask] + (randn(2 * npix) * sqrt(varmap)).reshape(2, npix)
        allcl.append(esti.get_spectra(dm, dm))

    figure()
    subplot(3, 1, 1)
    plot(lth, clth.transpose()[lth, 1: 3], '--k')
    hcl = mean(allcl, 0).transpose()
    scl = std(allcl, 0).transpose()
    plot(ellval, hcl, 'b.')
    plot(ellval, hcl + scl, 'r--', label=r"$\pm 1\sigma$")
    plot(ellval, hcl - scl, 'r--')
    ylabel(r"$C_\ell$")
    semilogy()
    legend(loc=4)
    subplot(3, 1, 2)
    cosmic = sqrt(2./(2 * lth + 1)) / mean(mask) * clth[1: 3, lth]
    plot(lth, cosmic.transpose(), '--k')
    plot(ellval, scl, 'r-', label=r"$\sigma_{\rm MC}$")
    plot(ellval, sqrt(diag(V)).reshape(nder, -1).transpose(), 'b.')
    ylabel(r"$\sigma(C_\ell)$")
    semilogy()
    legend(loc=4)
    subplot(3, 1, 3)
    plot(ellval, (hcl-clth[1: 3, lth].T)/(scl/sqrt(nsimu)), 'o')
    ylabel(r"$R[C_\ell]$")
    xlabel(r"$\ell$")
    ylim(-3, 3)
    show()


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
