import os
import healpy as hp
import numpy as np
import unittest

nside=8
lmax=10
tagnames = ['TT','EE','BB','TE','TB','EB']
muKarcmin = 0.1
nstoke = 2

#read init spectrum
clth = hp.read_cl( "%s/../../examples/planck_base_planck_2015_TTlowP.fits" % os.path.dirname(__file__))

clout = np.array([[ 2.03788961e-14,  3.94846164e-14,  1.70567361e-14,
         1.70038780e-14,  7.23238986e-15,  4.12350738e-15,
         1.15917385e-15,  7.30325095e-16,  4.40443054e-16],
       [ 2.77539367e-15,  2.32034489e-15,  5.79381145e-16,
         7.07583749e-16,  6.85834147e-16,  4.10994411e-16,
         2.94599123e-16,  1.27743335e-16,  7.60353819e-17],
       [-1.13214800e-15,  1.74772986e-15, -3.67311693e-16,
         1.29416072e-15, -3.07591887e-16,  9.44383890e-17,
         5.53306003e-17,  7.34264668e-18,  2.39544262e-17]])


class xqmlTest( unittest.TestCase):
    def test_xqml(self):
        from xqml import xQML, Bins
        from xqml.simulation import Karcmin2var
        fwhm = 1.
        
        #create map
        np.random.seed( 1234)
        mapCMB = hp.synfast( clth, nside, new=True, fwhm=np.deg2rad(fwhm), pixwin=True, verbose=False)

        #create mask
        t,p=hp.pix2ang( nside, range(hp.nside2npix(nside)))
        mask = np.ones(hp.nside2npix(nside), bool)
        mask[np.abs(90-np.rad2deg(t)) < 10] = 0.
        npix = sum(mask)

        #generate binning from l=2 to lmax with deltal=1
        binning = Bins.fromdeltal( 2, lmax, 1)

        #noise map
        pixvar = Karcmin2var(muKarcmin*1e-6, nside)
        varmap = np.ones((nstoke * npix)) * pixvar
        NoiseVar = np.diag(varmap)
        mapN = np.random.randn(nstoke, npix) * np.sqrt(pixvar)

        #generate xpol class
        spec = ["EE","BB","EB"]
        xq = xQML(mask, binning, clth, NA=NoiseVar, lmax=lmax, fwhm=fwhm, spec=spec)

        #compute spectra from map dT
        cl = xq.get_spectra( mapCMB[1:,mask]+mapN)

        print( "clout:", clout[0,0:4])
        print( "cl:", cl[0,0:4])

        np.testing.assert_almost_equal(cl,clout)
