from pylab import *
import astropy.io.fits as fits
import xqml
import timeit
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt


nside = 8
lmax = 3*nside-1
Slmax = 3*nside-1
deltal = 1
clth = hp.read_cl( 'planck_base_planck_2015_TTlowP.fits')
lth = arange(2,lmax+1)

ellbins = arange(2,lmax+2,deltal)
ellbins[-1] = lmax+1

P, Q, ell, ellval = xqml.GetBinningMatrix(ellbins, lmax)
nbins=len(ellbins)-1


#create mask
t,p=hp.pix2ang( nside, range(hp.nside2npix(nside)))
mask = np.ones(hp.nside2npix(nside),bool)
mask[abs(90-rad2deg(t)) < 10] = False
npix = sum(mask)

fwhm = 0.5
bl=hp.gauss_beam(deg2rad(fwhm), lmax=Slmax+1)

allStoke, der, ind = xqml.getstokes(polar=True,temp=False,EBTB=False)
nder=len(der)

muKarcmin = 0.1


pixvar = 2*xqml.muKarcmin2var(muKarcmin, nside)
varmap = ones((2*npix))*pixvar
NoiseVar = np.diag(varmap)

cmb = hp.synfast( clth, nside, fwhm=deg2rad(fwhm), pixwin=True, new=True, verbose=False)
noise = (randn(len(varmap)) * varmap**.5).reshape(2,-1)
dm = cmb[1:,mask] + noise


###################################### Compute ds_dcb ######################################
ip=arange(hp.nside2npix(nside))
ipok=ip[mask]

Pl, S = xqml.compute_ds_dcb(ellbins,nside,ipok,bl,clth,Slmax,polar=True,temp=False, EBTB=False, pixwining=True, timing=True, MC=False)
Pl = Pl.reshape((nder)*(np.shape(Pl)[1]), 2*npix, 2*npix)






###################################### Compute spectra ######################################
esti = xqml.xQML(mask,ellbins, clth, Pl=Pl, fwhm=fwhm)
esti.construct_esti( NoiseVar, NoiseVar)
cl = esti.get_spectra( dm, dm)
V = esti.get_covariance()


###################################### Construct MC ######################################
allcl = []
esti = xqml.xQML(mask,ellbins, clth, Pl=Pl, fwhm=fwhm)
esti.construct_esti( NoiseVar, NoiseVar)
start = timeit.default_timer()
for n in np.arange(100):
	xqml.progress_bar( n, 100,timeit.default_timer()-start)
	cmb = hp.synfast( clth, nside, fwhm=deg2rad(fwhm), pixwin=True, new=True, verbose=False)
	dm = cmb[1:,mask] + (randn(2*npix)*sqrt(varmap)).reshape( 2, npix)
	allcl.append(esti.get_spectra( dm, dm))


figure()
subplot( 2,1,1)
plot( lth, clth.transpose()[lth,1:3], 'k')
plot( ellval, mean( allcl,0).transpose(), 'r')
plot( ellval, mean( allcl,0).transpose() + std( allcl,0).transpose(), 'r--')
plot( ellval, mean( allcl,0).transpose() - std( allcl,0).transpose(), 'r--')
semilogy()
subplot( 2,1,2)
cosmic = sqrt( 2./(2*lth+1))/mean(mask) * clth[1:3,lth]
plot( lth, cosmic.transpose(), 'k')
plot( ellval, std( allcl,0).transpose(), 'r')
plot( ellval, sqrt(diag(V)).reshape(nder,-1).transpose(), 'b')
semilogy()




show()
