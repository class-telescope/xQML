"""
(Cross-) power spectra estimation using the QML method.
[Vanneste et al. 2018, arXiv:1807.02484]
"""
from __future__ import division

import sys
import timeit
import string

from scipy import linalg

import numpy as np
import healpy as hp
import random as rd

from .bins import Bins

from .xqml_utils import getstokes, pd_inv, GetBinningMatrix

from .estimators import El
from .estimators import CovAB
from .estimators import CrossGisherMatrix
from .estimators import CrossWindowFunction, CrossWindowFunctionLong
from .estimators import yQuadEstimator, ClQuadEstimator
from .estimators import biasQuadEstimator

from .libcov import compute_ds_dcb, S_bins_MC, compute_S, covth_bins_MC, compute_PlS, SignalCovMatrix


__all__ = ['xQML', 'Bins']


class xQML(object):
    """ Main class to handle the spectrum estimation """
    def __init__(self, mask, bins, clth, NA=None, NB=None, lmax=None, Pl=None,
                  S=None, fwhm=0., bell=None, spec=['EE','BB'], pixwin=True, verbose=True):
        """
        Parameters
        ----------
        mask : 1D array of booleans
            Mask defining the region of interest (of value True)
        bins : Bins class object
            Contains informations about bins
        clth : ndarray of floats
            Array containing fiducial CMB spectra (unbinned)
        lmax : int
            Maximum multipole
        Pl : ndarray or None, optional
            Normalize Legendre polynomials dS/dCl. Default: None
        fwhm : float, optional
            FWHM of the experiment beam in degree
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
        
        # binning (Bins class)
        self.bins = bins
        
        # Maximum multipole based on nside (rule of thumb to avoid aliasing)
        self.Slmax = bins.lmax if lmax is None else lmax
        
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

        nbin = bins.nbins
        nmem = self.nspec*nbin*(self.nstokes*self.npix)**2
        toGb = 1024. * 1024. * 1024.
        if verbose:
            print("xQML")
            print("spec: ", spec)
            print("nbin: ", nbin)
            print("Memset: %.2f Gb (%d,%d,%d,%d)" % (8.*nmem/toGb,self.nspec,nbin,self.nstokes,self.npix))
        
        # If Pl is given by the user, just load it, and then compute the signal
        # covariance using the fiducial model.
        # Otherwise compute Pl and S from the arguments.
        # Ok, but Pl cannot be binned, otherwise S construction is not valid
        if Pl is None:
            self.Pl, self.S = compute_ds_dcb(self.bins,
                                             self.nside,
                                             self.ipok,
                                             self.bl,
                                             clth,
                                             self.Slmax,
                                             self.spec,
                                             pixwin=self.pixwin,
                                             verbose=verbose,
                                             openMP=True)
        else:
            self.Pl = Pl
            if S is None:
                self.S = self._SignalCovMatrix(clth)
            else:
                self.S = S
        
        if NA is not None:
            self.construct_esti(NA=NA, NB=NB, verbose=verbose, thread=True)

    def construct_esti(self, NA, NB, verbose=False, thread=True):
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
        thread: bool=True
            do OMP or threading
        """

        tstart = timeit.default_timer()
        
        # Invert (signalA + noise) matrix
        invCa = pd_inv(self.S + NA)

        # Invert (signalB + noise) matrix
        invCb = pd_inv(self.S + NB)
        
        # Compute El = Ca^-1.Pl.Cb^-1 (long)
        self.El = El(invCa, invCb, self.Pl, openMP=True, thread=thread, verbose=verbose)
        
        # Finally compute invW by inverting (longer)
        self.invW = linalg.inv(CrossWindowFunction(self.El, self.Pl, openMP=True, thread=thread, verbose=verbose))
        
        # Compute bias for auto
        if not self.cross:
            self.bias = biasQuadEstimator(NA, self.El)
        # self.bias = biasQuadEstimator(self.NA, self.El)
        
        if verbose:
            print("Construct estimator: %.1f sec" % (timeit.default_timer()-tstart))

    def __call__(self, mapA, mapB=None):
        """
        Return the unbiased spectra
        Parameters
        ----------
        mapA, mapB : 1D array
            Pixel map number 1/2. The maps should have shape (3, npix) or (nstoeks*npix_masked), in the former case, the
            masking will be applied to the maps.
        Returns
        ----------
        cl : array or sequence of arrays
            Returns cl or a list of cl's (TT, EE, BB, TE, TB, EB)
        """
        # Define conditions based on the map size
        if self.cross:
            assert mapB is not mapA, "can't use the same map for cross spectra."
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
        if self.cross:
            G = CrossGisherMatrix(self.El, self.S)
        else:
            G = CrossGisherMatrix(self.El, self.S + self.NA)

        # # Do V = W^-1.G.W^-1 + W^-1
        V = CovAB(self.invW, G)
        return V

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
        return SignalCovMatrix(self.Pl, model)
