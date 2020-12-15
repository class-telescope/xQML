#!/usr/bin/env python
"""
Test script for xQML

Author: Vanneste
"""

from __future__ import division

import numpy as np
import healpy as hp
from pylab import *
import astropy.io.fits as fits
import timeit
import sys

import xqml
from xqml.xqml_utils import progress_bar, getstokes
from xqml.simulation import Karcmin2var
from xqml.simulation import extrapolpixwin
ion()
show()

exp = "Big"
if len(sys.argv) > 1:
    if sys.argv[1].lower()[0] == "s":
        exp = "Small"

if exp == "Big":
    nside = 16
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
nsimu = 10
MODELFILE = 'planck_base_planck_2015_TTlowP.fits'
Slmax = lmax

s0 = timeit.default_timer()

# provide list of specs to be computed, and/or options
spec = ['EE','BB'] #'EB', 'TE', 'TB']
pixwin = True
ellbins = np.arange(2, lmax + 2, dell)
ellbins[-1] = lmax+1

muKarcmin = 1.0
fwhm = 0.5



##############################
#input model
clth = np.array(hp.read_cl(MODELFILE))
Clthshape = zeros(((6,)+shape(clth)[1:]))
Clthshape[:4] = clth
clth = Clthshape
lth = arange(2, lmax+1)
##############################



##############################
# Create mask

t, p = hp.pix2ang(nside, range(hp.nside2npix(nside)))
mask = np.ones(hp.nside2npix(nside), bool)
# import random
# random.shuffle(mask)

if exp == "Big":
#    mask[abs(90 - rad2deg(t)) < glat] = False
    mask[(90 - rad2deg(t)) < glat] = False
elif exp == "Small":
    mask[(90 - rad2deg(t)) < glat] = False

fsky = np.mean(mask)
npix = sum(mask)
print("fsky=%.2g %% (npix=%d)" % (100*fsky,npix))
toGB = 1024. * 1024. * 1024.
emem = 8.*(npix*2*npix*2) * ( len(lth)*2 ) / toGB
print("mem=%.2g Gb" % 2.*emem)
##############################



stokes, spec, istokes, ispecs = getstokes( spec=spec)
print(stokes, spec, istokes, ispecs)
nspec = len(spec)
nstoke = len(stokes)


# ############## Generate Noise ###############
pixvar = Karcmin2var(muKarcmin*1e-6, nside)
varmap = ones((nstoke * npix)) * pixvar
NoiseVar = np.diag(varmap)

noise = (randn(len(varmap)) * varmap**0.5).reshape(nstoke, -1)



# ############## Initialise xqml class ###############
esti = xqml.xQML(mask, ellbins, clth, lmax=lmax, fwhm=fwhm, spec=spec)
s1 = timeit.default_timer()
print( "Init: %.2f sec (%.2f)" % (s1-s0,s1-s0))

esti.NA = NoiseVar
esti.NB = NoiseVar

invCa = xqml.xqml_utils.pd_inv(esti.S + esti.NA)
invCb = xqml.xqml_utils.pd_inv(esti.S + esti.NB)
s2 = timeit.default_timer()
print( "Inv C: %.2f sec (%.2f)" % (s2-s0,s2-s1))
s1 = s2

meth = "classic"
#meth = "long"

if meth == "classic":
    esti.El = xqml.estimators.El(invCa, invCb, esti.Pl)
    s2 = timeit.default_timer()
    print( "Construct El: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1 = s2

    Wll = xqml.estimators.CrossWindowFunction(esti.El, esti.Pl)
#    nl = len(esti.El)
#    Wll = np.asarray( [np.sum(E * P) for E in esti.El for P in esti.Pl] ).reshape(nl,nl)
    s2 = timeit.default_timer()
    print( "Construct W: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1=s2
    esti.Pl = 0.

    esti.bias = xqml.estimators.biasQuadEstimator(esti.NA, esti.El)
    s2 = timeit.default_timer()
    print( "Construct bias: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1 = s2

else:
    nl = len(esti.Pl)
    CaPl = [np.dot(invCa, P) for P in esti.Pl]
    CbPl = [np.dot(invCb, P) for P in esti.Pl]
    esti.Pl = 0
    Wll = np.asarray([np.sum(CaP * CbP) for CaP in CaPl for CbP in CbPl]).reshape(nl,nl)
    s2 = timeit.default_timer()
    print( "Construct Wll: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1 = s2
    CbPl = 0

    esti.El = [np.dot(CaP, invCb) for CaP in CaPl]
#    esti.El = xqml.estimators.El(invCa, invCb, esti.Pl)
    s2 = timeit.default_timer()
    print( "Construct El: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1 = s2
    CaPl =0

    esti.bias = xqml.estimators.biasQuadEstimator(esti.NA, esti.El)
    s2 = timeit.default_timer()
    print( "Construct bias: %.2f sec (%.2f)" % (s2-s0,s2-s1))
    s1 = s2


esti.invW = linalg.inv(Wll)
s2 = timeit.default_timer()
print( "inv W: %.2f sec (%.2f)" % (s2-s0,s2-s1))
s1=s2

#esti.construct_esti( NA=NoiseVar, NB=NoiseVar)
#s2 = timeit.default_timer()
#print( "Construct esti: %.2f sec (%.2f)" % (s2-s0,s2-s1))
ellval = esti.lbin()


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

print( "get_spectra: %.2f (%.2f sec)" % (timeit.default_timer()-s0,mean(t)))
hcl = mean(allcl, 0)
scl = std(allcl, 0)
hcla = mean(allcla, 0)
scla = std(allcla, 0)





## if __name__ == "__main__":
##     """
##     Run the doctest using

##     python simulation.py

##     If the tests are OK, the script should exit gracefuly, otherwise the
##     failure(s) will be printed out.
##     """
##     import doctest
##     if np.__version__ >= "1.14.0":
##         np.set_printoptions(legacy="1.13")
##     doctest.testmod()
