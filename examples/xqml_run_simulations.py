#!/usr/bin/env python
"""
Test script for xQML
"""

from __future__ import division

import numpy as np
import healpy as hp
from pylab import *
import timeit
import sys

import xqml
from xqml.xqml_utils import progress_bar, getstokes
from xqml.simulation import Karcmin2var
from xqml.simulation import extrapolpixwin
#ion()
#show()

patch = "Big"
if len(sys.argv) > 1:
    if sys.argv[1].lower()[0] == "s":
        patch = "Small"

if patch == "Big":
    nside = 8
    dell = 1
    glat = 10
    fwhm = 1 #deg
    lmin=2
elif patch == "Small":
    nside = 64
    dell = 10
    glat = 70
    fwhm = 1 #deg
    lmin = 2
else:
    print( "Need a patch !")

#lmax = nside
lmax = 3 * nside - 1
nsimu = 100
MODELFILE = 'planck_base_planck_2015_TTlowP.fits'

# provide list of specs to be computed, and/or options
spec = ['EE','BB','EB']
pixwin = True

muKarcmin = 0.1



##############################
#input model
clth = np.array(hp.read_cl(MODELFILE))
clth = array( list(clth) + list(clth[0:2]*0.))
lth = arange(2, lmax+1)
##############################



##############################
# Create mask
t, p = hp.pix2ang(nside, range(hp.nside2npix(nside)))
mask = np.ones(hp.nside2npix(nside), bool)
# import random
# random.shuffle(mask)

if patch == "Big":
    mask[abs(90 - rad2deg(t)) < glat] = False
elif patch == "Small":
    mask[(90 - rad2deg(t)) < glat] = False

fsky = np.mean(mask)
npix = sum(mask)
print("%s patch: fsky=%.2g %% (npix=%d)" % (patch,100*fsky,npix))
toGB = 1024. * 1024. * 1024.
emem = 8.*(npix*2*npix*2) * ( len(lth)*2 ) / toGB
print("mem=%.2g Gb" % emem)
##############################



stokes, spec, istokes, ispecs = getstokes( spec=spec)
print(stokes, spec, istokes, ispecs)
nspec = len(spec)
nstoke = len(stokes)


# ############## Generate White Noise ###############
pixvar = Karcmin2var(muKarcmin*1e-6, nside)
varmap = ones((nstoke * npix)) * pixvar
NoiseVar = np.diag(varmap)


# ############## Initialise xqml class ###############
bins = xqml.Bins.fromdeltal( lmin, lmax, dell)
esti = xqml.xQML(mask, bins, clth, NA=NoiseVar, NB=NoiseVar, lmax=lmax, fwhm=fwhm, spec=spec)
lb = bins.lbin


# ############## Compute Analytical variance ###############
#V  = esti.get_covariance(cross=True )
#Va = esti.get_covariance(cross=False)
#s2 = timeit.default_timer()
#print( "construct covariance: %d sec" % (s2-s1))


# ############## Construct MC ###############
allcla = []
allcl = []
t = []
bl = hp.gauss_beam(deg2rad(fwhm), lmax=lmax)
fpixw = extrapolpixwin(nside, lmax, pixwin=pixwin)
for n in range(nsimu):
    progress_bar(n, nsimu)
    cmb = np.array(hp.synfast(clth[:, :len(fpixw)]*(fpixw*bl)**2, nside, pixwin=False, lmax=lmax, fwhm=0.0, new=True, verbose=False))
    cmbm = cmb[istokes][:, mask]
    dmA = cmbm + np.random.randn(nstoke, npix) * sqrt(pixvar)
    dmB = cmbm + np.random.randn(nstoke, npix) * sqrt(pixvar)
    s1 = timeit.default_timer()
    allcl.append(esti.get_spectra(dmA, dmB))
    t.append( timeit.default_timer() - s1)
    allcla.append(esti.get_spectra(dmA))

print( "get_spectra: %f sec" % mean(t))
hcl = mean(allcl, 0)
scl = std(allcl, 0)
hcla = mean(allcla, 0)
scla = std(allcla, 0)



# ############## Plot results ###############

figure(figsize=[12, 8])
toDl = lb*(lb+1)/2./np.pi

subplot(3, 2, 1)
title("Cross")
plot(lth, (lth*(lth+1)/2./np.pi)[:, None]*clth[ispecs][:, lth].T, '--k')
for s in np.arange(nspec):
    errorbar(lb, toDl*hcl[s], yerr=toDl*scl[s], xerr=bins.dl/2, fmt='o', color='C%i' % ispecs[s], label=r"$%s$" % spec[s])
semilogy()
ylabel(r"$D_\ell$")
legend(loc=4, frameon=True)

subplot(3, 2, 2)
title("Auto")
plot(lth,(lth*(lth+1)/2./np.pi)[:, None]*clth[ispecs][:, lth].T, '--k')
for s in np.arange(nspec):
    errorbar(lb, toDl*hcla[s], yerr=toDl*scla[s], xerr=bins.dl/2, fmt='o', color='C%i' % ispecs[s], label=r"$%s$" % spec[s])
semilogy()

subplot(3, 2, 3)
for s in np.arange(nspec):
    plot(lb, toDl*scl[s], color='C%i' % ispecs[s], label=r"$\sigma^{%s}_{\rm MC}$" % spec[s])
#    plot(lb, sqrt(diag(V)).reshape(nspec, -1)[s], '--', color='C%i' % ispecs[s])
ylabel(r"$\sigma(D_\ell)$")
semilogy()

subplot(3, 2, 4)
for s in np.arange(nspec):
    plot(lb, toDl*scla[s], color='C%i' % ispecs[s], label=r"$\sigma^{%s}_{\rm MC}$" % spec[s])
#    plot(lb, sqrt(diag(Va)).reshape(nspec, -1)[s], '--', color='C%i' % ispecs[s])
semilogy()

subplot(3, 2, 5)
for s in np.arange(nspec):
    plot(lb, (hcl[s]-bins.bin_spectra(clth)[ispecs[s]])/(scl[s]/sqrt(nsimu)), '--o', color='C%i' % ispecs[s])
ylabel(r"$R[C_\ell]$")
xlabel(r"$\ell$")
ylim(-3, 3)
grid()

subplot(3, 2, 6)
for s in np.arange(nspec):
    plot(lb, (hcla[s]-bins.bin_spectra(clth)[ispecs[s]])/(scla[s]/sqrt(nsimu)), '--o', color='C%i' % ispecs[s])
xlabel(r"$\ell$")
ylim(-3, 3)
grid()

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
