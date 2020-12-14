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

exp = "Big"
if len(sys.argv) > 1:
    if sys.argv[1].lower()[0] == "s":
        exp = "Small"

if exp == "Big":
    nside = 8
    dell = 1
    glat = 10
elif exp == "Small":
    nside = 64
    dell = 10
    glat = 80
else:
    print( "Need a patch !")

#lmax = nside
lmax = 2 * nside - 1
nsimu = 100
MODELFILE = 'planck_base_planck_2015_TTlowP.fits'
Slmax = 3*nside-1


# provide list of specs to be computed, and/or options
spec = ['EE','BB','EB']
#spec = ['TT','EE','BB','TE']#'EE','BB', 'EB']#, 'TE', 'TB']
pixwin = True
ellbins = np.arange(2, lmax + 2, dell)
ellbins[-1] = lmax+1

muKarcmin = 1.0
fwhm = 10



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

if exp == "Big":
    mask[abs(90 - rad2deg(t)) < glat] = False
elif exp == "Small":
    mask[(90 - rad2deg(t)) < glat] = False

fsky = np.mean(mask)
npix = sum(mask)
print("%s patch: fsky=%.2g %% (npix=%d)" % (exp,100*fsky,npix))
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
start = timeit.default_timer()
esti = xqml.xQML(mask, ellbins, clth, NA=NoiseVar, NB=NoiseVar, lmax=lmax, fwhm=fwhm, spec=spec)
s1 = timeit.default_timer()
print( "Init: %d sec" % (s1-start))
ellval = esti.lbin()

# ############## Compute Analytical variance ###############
#V  = esti.get_covariance(cross=True )
#Va = esti.get_covariance(cross=False)
s2 = timeit.default_timer()
print( "construct covariance: %d sec" % (s2-s1))


# ############## Construct MC ###############
allcla = []
allcl = []
t = []
bl = hp.gauss_beam(deg2rad(fwhm), lmax=Slmax)
fpixw = extrapolpixwin(nside, Slmax, pixwin=pixwin)
for n in np.arange(nsimu):
    progress_bar(n, nsimu)
    cmb = np.array(hp.synfast(clth[:, :len(fpixw)]*(fpixw*bl)**2, nside,
                   pixwin=False, lmax=Slmax, fwhm=0.0, new=True, verbose=False))
    cmbm = cmb[istokes][:, mask]
    dmA = cmbm + (randn(nstoke * npix) * sqrt(varmap)).reshape(nstoke, npix)
    dmB = cmbm + (randn(nstoke * npix) * sqrt(varmap)).reshape(nstoke, npix)
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
clf()
Delta = (ellbins[1:] - ellbins[:-1])/2.

subplot(3, 2, 1)
title("Cross")
plot(lth, (lth*(lth+1)/2./np.pi)[:, None]*clth[ispecs][:, lth].T, '--k')
for s in np.arange(nspec):
    errorbar(ellval, ellval*(ellval+1)/2./np.pi*hcl[s], yerr=scl[s], xerr=Delta, fmt='o', color='C%i' % ispecs[s], label=r"$%s$" % spec[s])
semilogy()
ylabel(r"$D_\ell$")
legend(loc=4, frameon=True)

subplot(3, 2, 2)
title("Auto")
plot(lth,(lth*(lth+1)/2./np.pi)[:, None]*clth[ispecs][:, lth].T, '--k')
for s in np.arange(nspec):
    errorbar(ellval, ellval*(ellval+1)/2./np.pi*hcla[s], yerr=scla[s], xerr=Delta, fmt='o', color='C%i' % ispecs[s], label=r"$%s$" % spec[s])
semilogy()

subplot(3, 2, 3)
for s in np.arange(nspec):
    plot(ellval, scl[s], 'o', color='C%i' % ispecs[s], label=r"$\sigma^{%s}_{\rm MC}$" % spec[s])
#    plot(ellval, sqrt(diag(V)).reshape(nspec, -1)[s], '-', color='C%i' % ispecs[s])
ylabel(r"$\sigma(C_\ell)$")
semilogy()

subplot(3, 2, 4)
for s in np.arange(nspec):
    plot(ellval, scla[s], 'o', color='C%i' % ispecs[s], label=r"$\sigma^{%s}_{\rm MC}$" % spec[s])
#    plot(ellval, sqrt(diag(Va)).reshape(nspec, -1)[s], '-', color='C%i' % ispecs[s])
semilogy()

subplot(3, 2, 5)
for s in np.arange(nspec):
    plot(ellval, (hcl[s]-esti.BinSpectra(clth)[s])/(scl[s]/sqrt(nsimu)), '--o', color='C%i' % ispecs[s])
ylabel(r"$R[C_\ell]$")
xlabel(r"$\ell$")
ylim(-3, 3)
grid()

subplot(3, 2, 6)
for s in np.arange(nspec):
    plot(ellval, (hcla[s]-esti.BinSpectra(clth)[s])/(scla[s]/sqrt(nsimu)), '--o', color='C%i' % ispecs[s])
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
