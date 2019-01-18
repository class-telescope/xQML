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

import xqml

ion()
show()

# if __name__ == "__main__":

# # Inputs
nside = 16
lmax = 2 * nside - 3
Slmax = 2 * nside - 3
dell = 1
nsimu = 100
clth = np.array(hp.read_cl('planck_base_planck_2015_TTlowP.fits'))
Clthshape = zeros(((6,)+shape(clth)[1:]))
Clthshape[:4] = clth
clth = Clthshape

# # spectra correlations level
EB = 0.0
clth[4] = EB*sqrt(clth[1]*clth[2])
TB = 0.0
clth[5] = TB*sqrt(clth[0]*clth[2])

# provide list of specs to be computed, and/or options
lth = arange(2, lmax+1)
spec = None # ['EB', 'TE', 'TB']
temp = False
polar = True
corr = False
pixwin = True

# ellbins = np.append(2, arange(50, lmax + 2, dell))
ellbins = np.arange(2, lmax + 2, dell)
ellbins[-1] = lmax + 1

P, Q, ell, ellval = xqml.simulation.GetBinningMatrix(ellbins, lmax)
nbins = len(ellbins) - 1

# Create mask
t, p = hp.pix2ang(nside, range(hp.nside2npix(nside)))
mask = np.ones(hp.nside2npix(nside), bool)

# # Large scale mask
# mask[abs(90 - rad2deg(t)) < 30] = False

# # Small scale mask (do not forget to change dell)
# mask[(90 - rad2deg(t)) < 78] = False

fsky = np.mean(mask)
print("fsky=%.2g %%" % (100*fsky))
npix = sum(mask)

fwhm = 0.5
# bl = hp.gauss_beam(deg2rad(fwhm), lmax=Slmax + 1)
bl = hp.gauss_beam(deg2rad(fwhm), lmax=Slmax)

stokes, spec, istokes, ispecs = xqml.xqml_utils.getstokes(
    spec=spec, polar=polar, temp=temp, corr=corr)
print(stokes, spec, istokes, ispecs)
nspec = len(spec)
nstoke = len(stokes)

# ############## Compute ds_dcb ###############
ip = arange(hp.nside2npix(nside))
ipok = ip[mask]

Pl, S = xqml.libcov.compute_ds_dcb(ellbins, nside, ipok, bl, clth, Slmax, spec=spec,
                              pixwin=pixwin, timing=True, MC=False)

# ############## Initialise xqml class ###############

muKarcmin = 1.0
pixvar = xqml.simulation.muKarcmin2var(muKarcmin, nside)
varmap = ones((nstoke * npix)) * pixvar
NoiseVar = np.diag(varmap)

noise = (randn(len(varmap)) * varmap**0.5).reshape(nstoke, -1)
esti = xqml.xQML(mask, ellbins, clth, NA=NoiseVar, NB=NoiseVar, Pl=Pl,
                 S=S, fwhm=fwhm, spec=spec, temp=temp, polar=polar, corr=corr)
# esti.construct_esti(NoiseVar, NoiseVar)
V = esti.get_covariance(cross=True)
Va = esti.get_covariance(cross=False)

# ############## Construct MC ###############

allcla = []
allcl = []
# allcmb = []
esti.construct_esti(NoiseVar, NoiseVar)
fpixw = xqml.simulation.extrapolpixwin(nside, Slmax+1, pixwin=pixwin)
start = timeit.default_timer()
for n in np.arange(nsimu):
    # progress_bar(n, nsimu, timeit.default_timer() - start)
    cmb = np.array(hp.synfast(clth[:, :len(fpixw)]*(fpixw*bl)**2, nside,
                   pixwin=False, lmax=Slmax, fwhm=0.0, new=True,
                   verbose=False))
    cmbm = cmb[istokes][:, mask]
    # allcmb.append(cmbm)
    dmA = cmbm + (randn(nstoke * npix) * sqrt(varmap)).reshape(nstoke, npix)
    dmB = cmbm + (randn(nstoke * npix) * sqrt(varmap)).reshape(nstoke, npix)
    allcl.append(esti.get_spectra(dmA, dmB))
    allcla.append(esti.get_spectra(dmA))

# myS = cov(np.array(allcmb).reshape(nsimu, -1), rowvar=False)
hcl = mean(allcl, 0)
scl = std(allcl, 0)
hcla = mean(allcla, 0)
scla = std(allcla, 0)

figure(figsize=[12, 8])
clf()
Delta = (ellbins[1:] - ellbins[:-1])/2.
subplot(3, 2, 1)
title("Cross")
plot(lth, (lth*(lth+1)/2./np.pi)[:, None]*clth[ispecs][:, lth].T, '--k')
[errorbar(ellval, ellval*(ellval+1)/2./np.pi*hcl[s], yerr=scl[s], xerr=Delta, fmt='o', color='C%i' % ispecs[s], label=r"$%s$" % spec[s])
    for s in np.arange(nspec)]
semilogy()
ylabel(r"$D_\ell$")
legend(loc=4, frameon=True)

subplot(3, 2, 2)
title("Auto")
plot(lth,(lth*(lth+1)/2./np.pi)[:, None]*clth[ispecs][:, lth].T, '--k')
[errorbar(ellval, ellval*(ellval+1)/2./np.pi*hcla[s], yerr=scla[s], xerr=Delta, fmt='o', color='C%i' % ispecs[s], label=r"$%s$" %
          spec[s]) for s in np.arange(nspec)]
semilogy()


subplot(3, 2, 3)
# cosmic = sqrt(2./(2 * lth + 1)) / mean(mask) * clth[ispecs][:, lth]
# plot(lth, cosmic.transpose(), '--k')
[plot(ellval, scl[s], '--', color='C%i' % s, label=r"$\sigma^{%s}_{\rm MC}$" %
      spec[s]) for s in np.arange(nspec)]
[plot(ellval, sqrt(diag(V)).reshape(nspec, -1)[s], 'o', color='C%i' % ispecs[s])
    for s in np.arange(nspec)]
ylabel(r"$\sigma(C_\ell)$")
semilogy()
# legend(loc=4, frameon=True)

subplot(3, 2, 4)
[plot(ellval, scla[s], ':', color='C%i' % s, label=r"$\sigma^{%s}_{\rm MC}$" %
      spec[s]) for s in np.arange(nspec)]
[plot(ellval, sqrt(diag(Va)).reshape(nspec, -1)[s], 'o', color='C%i' % ispecs[s])
    for s in np.arange(nspec)]
semilogy()

subplot(3, 2, 5)
[plot(ellval, (hcl[s]-P.dot(clth[ispecs[s]][lth]))/(scl[s]/sqrt(nsimu)), '--o',
      color='C%i' % ispecs[s]) for s in np.arange(nspec)]
ylabel(r"$R[C_\ell]$")
xlabel(r"$\ell$")
ylim(-3, 3)
grid()

subplot(3, 2, 6)
[plot(ellval, (hcla[s]-P.dot(clth[ispecs[s]][lth]))/(scla[s]/sqrt(nsimu)), '--o',
      color='C%i' % ispecs[s]) for s in np.arange(nspec)]
xlabel(r"$\ell$")
ylim(-3, 3)
grid()

show()

savefig("../Plots/Git/Nside%i_dell%i_fsky%.3g_spec%s.png" %
        (nside, dell, fsky, "".join(spec)))

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
