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
import libcov
import simulation
from xqml_utils import progress_bar
from simulation import muKarcmin2var, GetBinningMatrix
from simulation import extrapolpixwin
ion()

# if __name__ == "__main__":
nside = 4
lmax = 2 * nside - 1
Slmax = 2 * nside - 1
deltal = 1
nsimu = 10000
clth = np.array(hp.read_cl('planck_base_planck_2015_TTlowP.fits'))
Clthshape = zeros(((6,)+shape(clth)[1:]))
Clthshape[:4] = clth
clth = Clthshape
EB = 0.5
clth[4] = EB*sqrt(clth[1]*clth[2])
TB = 0.5
clth[5] = TB*sqrt(clth[0]*clth[2])

lth = arange(2, lmax+1)
spec = ['EB', 'TE', 'TB']
temp = True
polar = True
corr = False
pixwin = False

ellbins = arange(2, lmax + 2, deltal)
ellbins[-1] = lmax + 1

P, Q, ell, ellval = GetBinningMatrix(ellbins, lmax)
nbins = len(ellbins) - 1

# Create mask
t, p = hp.pix2ang(nside, range(hp.nside2npix(nside)))
mask = np.ones(hp.nside2npix(nside), bool)
mask[abs(90 - rad2deg(t)) < 60] = False
npix = sum(mask)

fwhm = 0.5
bl = hp.gauss_beam(deg2rad(fwhm), lmax=Slmax + 1)

stokes, spec, istokes, ispecs = simulation.getstokes(
    spec=spec, polar=polar, temp=temp, corr=corr)
print(stokes, spec, istokes, ispecs)
nspec = len(spec)
nstoke = len(stokes)

# ############## Compute ds_dcb ###############
ip = arange(hp.nside2npix(nside))
ipok = ip[mask]

Pl, S = libcov.compute_ds_dcb(ellbins, nside, ipok, bl, clth, Slmax, spec=spec,
                              pixwin=pixwin, timing=True, MC=False)

# ############## Compute spectra ###############

muKarcmin = 1.0
pixvar = muKarcmin2var(muKarcmin, nside)
varmap = ones((nstoke * npix)) * pixvar
NoiseVar = np.diag(varmap)

cmb = np.array(hp.synfast(clth, nside, fwhm=deg2rad(fwhm), pixwin=pixwin,
               new=True, verbose=False, lmax=Slmax))
noise = (randn(len(varmap)) * varmap**0.5).reshape(nstoke, -1)
dm = cmb[istokes][:, mask] + noise

esti = xqml.xQML(mask, ellbins, clth, Pl=Pl, S=S, fwhm=fwhm,
                 spec=spec, temp=temp, polar=polar, corr=corr)
esti.construct_esti(NoiseVar, NoiseVar)
cl = esti.get_spectra(dm, dm)
V = esti.get_covariance()

# ############## Construct MC ###############
allcl = []
# allcmb = []
# esti = xqml.xQML(mask, ellbins, clth, Pl=Pl, fwhm=fwhm, spec=spec, temp=temp,
#                    polar=polar, corr=corr)
esti.construct_esti(NoiseVar, NoiseVar)
fpixw = extrapolpixwin(nside, lmax+2, pixwin=pixwin)
start = timeit.default_timer()
for n in np.arange(nsimu):
    progress_bar(n, nsimu, timeit.default_timer() - start)
    cmb = np.array(hp.synfast(clth[:, :len(fpixw)]*(fpixw*bl)**2, nside,
                   lmax=Slmax, fwhm=deg2rad(fwhm), new=True, verbose=False))
    cmbm = cmb[istokes][:, mask]
    dmA = cmbm + (randn(nstoke * npix) * sqrt(varmap)).reshape(nstoke, npix)
    dmB = cmbm + (randn(nstoke * npix) * sqrt(varmap)).reshape(nstoke, npix)
    # allcmb.append(cmbm)
    allcl.append(esti.get_spectra(dmA, dmB))

figure(figsize=[10, 8])
clf()
subplot(3, 1, 1)
plot(lth, clth[ispecs][:, lth].T, '--k')
hcl = mean(allcl, 0)
scl = std(allcl, 0)
[plot(ellval, hcl[s], 'o', color='C%i' % s, label=r"$%s$" % spec[s])
    for s in np.arange(nspec)]
[fill_between(ellval, (hcl - scl/sqrt(nsimu))[s], (hcl + scl/sqrt(nsimu))[s],
              color='C%i' % s, alpha=0.2) for s in np.arange(nspec)]
ylabel(r"$C_\ell$")
semilogy()
legend(loc=4)

subplot(3, 1, 2)
cosmic = sqrt(2./(2 * lth + 1)) / mean(mask) * clth[ispecs][:, lth]
# plot(lth, cosmic.transpose(), '--k')
[plot(ellval, scl[s], '--', color='C%i' % s, label=r"$\sigma^{%s}_{\rm MC}$" %
      spec[s]) for s in np.arange(nspec)]
[plot(ellval, sqrt(diag(V)).reshape(nspec, -1)[s], 'o', color='C%i' % s)
    for s in np.arange(nspec)]
ylabel(r"$\sigma(C_\ell)$")
semilogy()
legend(loc=4)

subplot(3, 1, 3)
[plot(ellval, (hcl[s]-clth[ispecs[s]][lth])/(scl[s]/sqrt(nsimu)), '--o',
      color='C%i' % s) for s in np.arange(nspec)]
ylabel(r"$R[C_\ell]$")
xlabel(r"$\ell$")
ylim(-3, 3)
grid()
show()

# savefig("../Plots/Git/"+"test0.png")

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
