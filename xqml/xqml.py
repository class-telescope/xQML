"""
(Cross-) power spectra estimation using the QML method.
[Vanneste et al. 2018, arXiv:1807.02484]

Author: Vanneste
"""
from __future__ import division

import sys
import timeit
import string

from scipy import linalg

import numpy as np
import healpy as hp
import random as rd

from .xqml_utils import getstokes, pd_inv, GetBinningMatrix

from .estimators import El
from .estimators import CovAB
from .estimators import CrossGisherMatrix
from .estimators import CrossWindowFunction, CrossWindowFunctionLong
from .estimators import yQuadEstimator, ClQuadEstimator
from .estimators import biasQuadEstimator

from .libcov import compute_ds_dcb, S_bins_MC, compute_S, covth_bins_MC, compute_PlS, SignalCovMatrix


__all__ = ['xQML','Bins']


class xQML(object):
    """ Main class to handle the spectrum estimation """
    def __init__( self, mask, bins, clth, NA=None, NB=None, lmax=None, Pl=None,
                  S=None, fwhm=0., bell=None, spec=['EE','BB'], pixwin=True, verbose=True):
        """
        Parameters
        ----------
        mask : 1D array of booleans
            Mask defining the region of interest (of value True)
        bins : 1D array of floats (nbin+1)
            lower multipole bin
        clth : ndarray of floats
            Array containing fiducial CMB spectra (unbinned)
        lmax : int
            Maximum multipole
        Pl : ndarray or None, optional
            Normalize Legendre polynomials dS/dCl. Default: None
        fwhm : float, optional
            FWHM of the experiment beam
        bell : ndarray, optional
            beam transfer function (priority over fwhm)
        pixwin : boolean, optional
            If True, applies pixel window function to spectra. Default: True
        """
        self.bias = None
        self.cross = NB is not None
        self.NA = NA
        self.NB = NB if self.cross else NA
        # Number of pixels in the mask
        # For example that would be good to have an assertion
        # on the mask size, just to check that it corresponds to a valid nside.
        npixtot = len(mask)
        
        # Map resolution (healpix)
        self.nside = hp.npix2nside(npixtot)
        # ipok are pixel indexes outside the mask
        self.mask = np.asarray(mask,bool)
        self.ipok = np.arange(npixtot)[self.mask]
        self.npix = len(self.ipok)
        
        # lower multipole bin
        self.ellbins = bins
        
        # Maximum multipole based on nside (rule of thumb to avoid aliasing)
        self.Slmax = np.max(bins)-1 if lmax is None else lmax
            
        # Beam 2pt function (Gaussian)
        self.bl = hp.gauss_beam(np.deg2rad(fwhm), lmax=self.Slmax)
        if bell is not None:
            self.bl = bell[:self.Slmax+1]
        
        # Set the Stokes parameters needed        
        self.stokes, self.spec, self.istokes, self.ispecs = getstokes(spec)
        self.nstokes = len(self.stokes)
        self.nspec = len(self.spec)
        self.pixwin = pixwin
        
        clth = np.asarray(clth)
        if len(clth) == 4:
            clth = np.concatenate((clth,clth[0:2]*0.))

        nbin = len(self.ellbins)
        nmem = self.nspec*nbin*(self.nstokes*self.npix)**2
        toGb = 1024. * 1024. * 1024.
        if verbose:
            print( "xQML")
            print( "spec: ", spec)
            print( "nbin: ", nbin)
            print( "Memset: %.2f Gb (%d,%d,%d,%d)" % (8.*nmem/toGb,self.nspec,nbin,self.nstokes,self.npix))
        
        # If Pl is given by the user, just load it, and then compute the signal
        # covariance using the fiducial model.
        # Otherwise compute Pl and S from the arguments.
        # Ok, but Pl cannot be binned, otherwise S construction is not valid
        if Pl is None:
            self.Pl, self.S = compute_ds_dcb(
                self.ellbins, self.nside, self.ipok,
                self.bl, clth, self.Slmax,
                self.spec, pixwin=self.pixwin, verbose=verbose, MC=False, openMP=True)
        else:
            self.Pl = Pl
            if S is None:
                self.S = self._SignalCovMatrix(clth)
            else:
                self.S = S
        
        if NA is not None:
            self.construct_esti(NA=NA, NB=NB, verbose=verbose)
    
    def compute_dSdC( self, clth, lmax=None, verbose=True, MC=False, openMP=True):
        if lmax is None:
            lmax = 2*self.nside-1   #Why ?
        
        self.Pl, self.S = compute_ds_dcb( self.ellbins, self.nside, self.ipok, self.bl, clth, lmax,
                                          self.spec, pixwin=self.pixwin, verbose=verbose, MC=MC, openMP=openMP)
        return( self.Pl, self.S)

    def construct_esti(self, NA, NB=None, verbose=False):
        """
        Compute the inverse of the datasets pixel covariance matrices C,
        the quadratic matrix parameter E, and inverse of the window
        (mode-mixing) matrix W.

        Parameters
        ----------
        NA : 2D array
            Noise covariance matrix of dataset A
        NB : 2D array
            Noise covariance matrix of dataset B

        """
        self.cross = NB is not None
        self.NA = NA
        self.NB = NB if self.cross else NA

        tstart = timeit.default_timer()
        
        # Invert (signalA + noise) matrix
        invCa = pd_inv(self.S + self.NA)

        # Invert (signalB + noise) matrix
        invCb = pd_inv(self.S + self.NB)

        # Compute El = Ca^-11.Pl.Cb^-1 (long)
        self.El = El(invCa, invCb, self.Pl)
        
        # Finally compute invW by inverting (long)
        self.invW = linalg.inv(CrossWindowFunction(self.El, self.Pl))
        
        # Compute bias for auto
#        if not self.cross:
#            self.bias = biasQuadEstimator(self.NA, self.El)
        self.bias = biasQuadEstimator(self.NA, self.El)

        if verbose:
            print( "Construct estimator: %.1f sec" % (timeit.default_timer()-tstart))


    def get_spectra(self, mapA, mapB=None):
        """
        Return the unbiased spectra

        Parameters
        ----------
        map1 : 1D array
            Pixel map number 1
        map2 : 2D array
            Pixel map number 2

        Returns
        ----------
        cl : array or sequence of arrays
            Returns cl or a list of cl's (TT, EE, BB, TE, EB, TB)
        """
        # Define conditions based on the map size
        self.cross = mapB is not None
        cond_sizeA = np.size(mapA) == self.nstokes * self.npix
        dA = mapA if cond_sizeA else mapA[self.istokes][:,self.mask]
        if self.cross:
            cond_sizeB = np.size(mapB) == self.nstokes * self.npix
            dB = mapB if cond_sizeB else mapB[self.istokes][:,self.mask]

            yl = yQuadEstimator(dA.ravel(), dB.ravel(), self.El)
        else:
            yl = yQuadEstimator(dA.ravel(), dA.ravel(), self.El) - self.bias

        cl = ClQuadEstimator(self.invW, yl)

        # Return the reshaped set of cls
        return cl.reshape(self.nspec, -1)

    def get_covariance(self, cross=None):
        """
        Returns the analytical covariance of the spectrum based on the fiducial
        spectra model and pixel noise matrix.

        Returns
        ----------
        V : 2D matrix array of floats
            Covariance matrix of the spectra

        """
        # # Do Gll' = S^-1.El.S^-1.El'
        cross = self.cross if cross is None else cross
        if cross:
            G = CrossGisherMatrix(self.El, self.S)
        else:
            G = CrossGisherMatrix(self.El, self.S + self.NA)

        # # Do V = W^-1.G.W^-1 + W^-1
        V = CovAB(self.invW, G)

        return(V)

    def _SignalCovMatrix(self, clth):
        """
        Compute correlation matrix S = sum_l Pl*Cl

        Parameters
        ----------
        clth : ndarray of floats
            Array containing fiducial CMB spectra (unbinned).
        """
        # Choose only needed spectra according to ispec, and truncate
        # the ell range according the bin range. Flatten (1D) the result.
        model = clth[self.ispecs][:,2:int(self.ellbins[-1])].ravel()

        # Return scalar product btw Pl and the fiducial spectra.
        return SignalCovMatrix( self.Pl, model)

    def BinSpectra( self, clth):
        lmax = int(self.ellbins[-1])-1
        P, Q, ell, ellval = GetBinningMatrix(self.ellbins, lmax)
        return( np.asarray([P.dot(clth[isp,2:lmax+1]) for isp in self.ispecs]))

    def lbin( self):
        P, Q, ell, ellval = GetBinningMatrix(self.ellbins, self.Slmax)
        return( ellval)







##Not USED YET
class Bins(object):
    """
        lmins : list of integers
            Lower bound of the bins
        lmaxs : list of integers
            Upper bound of the bins (not included)
    """
    def __init__( self, lmins, lmaxs):
        if not(len(lmins) == len(lmaxs)):
            raise ValueError('Incoherent inputs')

        cutfirst = np.where( lmaxs>2)[0]        
        self.lmins = lmins[cutfirst]
        self.lmaxs = lmaxs[cutfirst]
        if self.lmins[0] < 2:
            self.lmins[0] = 2

        self._derive_ext()
    
    @classmethod
    def fromdeltal( cls, lmin, lmax, delta_ell):
        nbins = (lmax - lmin + 1) // delta_ell
        lmins = lmin + np.arange(nbins) * delta_ell
        lmaxs = lmins + delta_ell
        return( cls( lmins, lmaxs))

    def _derive_ext( self):
        self.lmin = min(self.lmins)
        self.lmax = max(self.lmaxs)-1
        if self.lmin < 1:
            raise ValueError('Input lmin is less than 1.')
        if self.lmax < self.lmin:
            raise ValueError('Input lmax is less than lmin.')
        
        self.nbins = len(self.lmins)
        self.lbin = (self.lmins + self.lmaxs - 1) / 2
        self.dl   = (self.lmaxs - self.lmins)

    def bins(self):
        return (self.lmins,self.lmaxs)
    
    def cut_binning(self, lmin, lmax):
        sel = np.where( (self.lmins > lmin) & (self.lmaxs <= lmax+1) )[0]
        self.lmins = self.lmins[sel]
        self.lmaxs = self.lmaxs[sel]
        self._derive_ext()
    
    def _bin_operators(self):
        ell2 = np.arange(self.lmax+1)
        ell2 = ell2 * (ell2 + 1) / (2 * np.pi)
        p = np.zeros((self.nbins, self.lmax+1))
        q = np.zeros((self.lmax+1, self.nbins))
        
        for b, (a, z) in enumerate(zip(self.lmins, self.lmaxs)):
            p[b, a:z] = ell2[a:z] / (z - a)
            q[a:z, b] = 1 / ell2[a:z]
        
        return p, q

    def bin_spectra(self, spectra):
        """
        Average spectra in bins specified by lmin, lmax and delta_ell,
        weighted by `l(l+1)/2pi`.
        """
        spectra = np.asarray(spectra)
        minlmax = min([spectra.shape[-1] - 1,self.lmax])
        fact_binned = 2 * np.pi / (self.lbin * (self.lbin + 1))
        _p, _q = self._bin_operators()
        return np.dot(spectra[..., :minlmax+1], _p.T[:minlmax+1,...]) * fact_binned

