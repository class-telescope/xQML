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

from xqml_utils import getstokes, pd_inv

from estimators import El
from estimators import CovAB
from estimators import CrossGisherMatrix
from estimators import CrossWindowFunction
from estimators import yQuadEstimator, ClQuadEstimator
from estimators import biasQuadEstimator

from libcov import compute_ds_dcb, S_bins_MC, compute_S, covth_bins_MC, compute_PlS, SignalCovMatrix


__all__ = ['xQML']


class xQML(object):
    """ Main class to handle the spectrum estimation """
    def __init__( self, mask, bins, clth, NA=None, NB=None, lmax=None, Pl=None,
                  S=None, fwhm=0., bell=None, spec=None, temp=False, polar=True, corr=False,
                  pixwin=True):
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
        polar : boolean, optional
            Compute the polarisation part E and B. Default: True
        temp : boolean, optional
            Compute the temperature part T. Default: False
        corr : boolean, optional
            If True, compute TE, TB, EB spectra. Default: False
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
        self.mask = np.array(mask,bool)
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
        # For example that would be good to assert that the user
        # set at least polar or temp to True.

        self.stokes, self.spec, self.istokes, self.ispecs = getstokes(
            spec, temp, polar, corr)
        self.nstokes = len(self.stokes)
        self.nspec = len(self.spec)
        self.pixwin = pixwin

        # If Pl is given by the user, just load it, and then compute the signal
        # covariance using the fiducial model.
        # Otherwise compute Pl and S from the arguments.
        # Ok, but Pl cannot be binned, otherwise S construction is not valid

        if Pl is None:
            self.Pl, self.S = compute_ds_dcb(
                self.ellbins, self.nside, self.ipok,
                self.bl, clth, self.Slmax,
                self.spec, pixwin=self.pixwin, timing=True, MC=False, openMP=True)
        else:
            self.Pl = Pl
            if S is None:
                self.S = self._SignalCovMatrix(clth)
            else:
                self.S = S
        if S is not None:
            self.S = S
        
        if NA is not None:
            self.construct_esti(NA=NA, NB=NB)

    def compute_dSdC( self, clth, lmax=None, timing=True, MC=False, openMP=True):
        if lmax is None:
            lmax = 2*self.nside-1   #Why ?
        
        self.Pl, self.S = compute_ds_dcb( self.ellbins, self.nside, self.ipok, self.bl, clth, lmax,
                                          self.spec, pixwin=self.pixwin, timing=timing, MC=MC, openMP=openMP)
        return( self.Pl, self.S)

    def construct_esti(self, NA, NB=None):
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

        # Invert (signalA + noise) matrix
#        self.invCa = linalg.inv(self.S + self.NA)
        self.invCa = pd_inv(self.S + self.NA)

        # Invert (signalB + noise) matrix
#        self.invCb = linalg.inv(self.S + self.NB)
        self.invCb = pd_inv(self.S + self.NB)

        # Compute E using Eq...
        self.E = El(self.invCa, self.invCb, self.Pl)
        if not self.cross:
            self.bias = biasQuadEstimator(self.NA, self.E)
        # Finally compute invW by inverting...
        self.invW = linalg.inv(CrossWindowFunction(self.E, self.Pl))

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
        # Should compute auto-spectra if map2 == None ?
        # Define conditions based on the map size
        self.cross = mapB is not None
        cond_sizeA = np.size(mapA) == self.nstokes * self.npix
        dA = mapA if cond_sizeA else mapA[self.istokes][:,self.mask]
        if self.cross:
            cond_sizeB = np.size(mapB) == self.nstokes * self.npix
            dB = mapB if cond_sizeB else mapB[self.istokes][:,self.mask]

            yl = yQuadEstimator(dA.ravel(), dB.ravel(), self.E)
            cl = ClQuadEstimator(self.invW, yl)
        else:
            if self.bias is None:
                self.bias = biasQuadEstimator(self.NA, self.E)
            yl = yQuadEstimator(dA.ravel(), dA.ravel(), self.E) - self.bias
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
            G = CrossGisherMatrix(self.E, self.S)
        else:
            G = CrossGisherMatrix(self.E, self.S + self.NA)

        # # Do V = W^-1.G.W^-1 + W^-1
        V = CovAB(self.invW, G)

        return(V)

    def _SignalCovMatrix(self, clth):
        """
        Compute correlation matrix S = sum_l Pl*Cl

        QUESTION: why you need to pass clth while it is in the constructor of
        the class already? you could just register it as
        an attribute (self.clth) and call it here.

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
