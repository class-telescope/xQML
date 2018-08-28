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

from libcov import compute_ds_dcb

from estimators import El
from estimators import CovAB
from estimators import CrossGisherMatrix
from estimators import CrossWindowFunction
from estimators import yQuadEstimator, ClQuadEstimator


class xQML(object):
    """ Main class to handle the spectrum estimation """
    def __init__(
            self, mask, bins, clth, lmax=None, Pl=None, S=None,
            fwhm=0., polar=True, temp=False, EBTB=False):
        """
        Parameters
        ----------
        mask : 1D array of booleans
            Mask defining the region of interest (of value True)
        bins : 1D array of floats
            Bin centers or bin edges?
        clth : ndarray of floats
            Array containing fiducial CMB spectra (unbinned)
        lmax : int
            Maximum multipole
        Pl : ndarray or None, optional
            Normalize Legendre polynomials dS/dCl. Default: None
        fwhm : float, optional
            FWHM of the experiment beam
        polar : boolean, optional
            Compute the polarisation part E and B. Default: True
        temp : boolean, optional
            Compute the temperature part T. Default: False
        EBTB : boolean, optional
            If True, compute EB and TB spectra. Default: False

        """
        # Number of pixels in the mask
        # For example that would be good to have an assertion
        # on the mask size, just to check that it corresponds to a valid nside.
        npix = len(mask)

        # Map resolution (healpix)
        self.nside = hp.npix2nside(npix)

        # ipok is...
        self.ipok = np.arange(npix)[np.array(mask, bool)]

        # Hum, you are asking for trouble as you have npix and self.npix
        # which are potentially not representing the same thing.
        self.npix = len(self.ipok)

        # Bin centers or edges?
        self.ellbins = bins

        # Maximum multipole based on nside (rule of thumb to avoid aliasing)
        self.Slmax = 3 * self.nside - 1 if lmax is None else lmax

        # Beam 2pt function (Gaussian)
        self.bl = hp.gauss_beam(np.deg2rad(fwhm), lmax=self.Slmax+1)

        # Set the Stokes parameters needed
        # For example that would be good to assert that the user
        # set at least polar or temp to True.
        self.stokes, self.istokes = self._getstokes(polar=polar, temp=temp)
        self.spec, self.ispec = self._getspec(
            polar=polar, temp=temp, EBTB=EBTB)
        self.nstokes = len(self.stokes)
        self.nspec = len(self.spec)

        # If Pl is given by the user, just load it, and then compute the signal
        # covariance using the fiducial model.
        # Otherwise compute Pl and S from the arguments.
        # Ok, but Pl cannot be binned, otherwise S construction is not valid
        if Pl is None:
            self.Pl, self.S = compute_ds_dcb(
                self.ellbins, self.nside, self.ipok,
                self.bl, clth, self.Slmax,
                polar=polar, temp=temp, EBTB=EBTB,
                pixwining=True, timing=True, MC=False)
        else:
            self.Pl = Pl
            if S is None:
                self.S = self._CorrelationMatrix(clth)
            else:
                self.S = S
        if S is not None:
            self.S = S

        # Reshape Pl into a square matrix (2D array)
        # self.Pl = self.Pl.reshape(
        # -1, self.nstokes * self.npix, self.nstokes * self.npix)

    def construct_esti(self, NA, NB):
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
        # Invert (signalA + noise) matrix
        self.invCa = linalg.inv(self.S + NA)

        # Invert (signalB + noise) matrix
        self.invCb = linalg.inv(self.S + NB)

        # Compute E using Eq...
        self.E = El(self.invCa, self.invCb, self.Pl)

        # Finally compute invW by inverting...
        self.invW = linalg.inv(CrossWindowFunction(self.E, self.Pl))

    def get_spectra(self, map1, map2):
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
        cond_size1 = np.size(map1) == self.nstokes * self.npix
        cond_size2 = np.size(map2) == self.nstokes * self.npix

        d1 = map1 if cond_size1 else map1[self.istokes, self.mask]
        d2 = map2 if cond_size2 else map2[self.istokes, self.mask]

        # yl is...
        yl = yQuadEstimator(d1.ravel(), d2.ravel(), self.E)

        # cl is obtained using...
        cl = ClQuadEstimator(self.invW, yl)

        # Return the reshaped set of cls
        return cl.reshape(self.nspec, -1)

    def get_covariance(self):
        """
        Returns the analytical covariance of the spectrum based on the fiducial
        spectra model and pixel noise matrix.

        Returns
        ----------
        V : 2D matrix array of floats
            Covariance matrix of the spectra

        """
        # # Do Gll' = S^-1.El.S^-1.El'
        G = CrossGisherMatrix(self.E, self.S)

        # # Do V = W^-1.G.W^-1 + W^-1
        V = CovAB(self.invW, G)

        return(V)

    def _getstokes(self, polar=True, temp=False):
        """
        Get the Stokes parameters number and name(s)

        Parameters
        ----------
        polar : boolean, optional
            Append Q, U to the list of Stokes parameters. Default: True
        temp : boolean, optional
            Append T to the list of Stokes parameters. Default: False

        Returns
        ----------
        stokes : list of strings
            List containing I or/and Q, U.
        indices : list of integers
            List containing position of I, Q and U in the
            stokes list (if present).
        """
        stokes = []
        if temp:
            stokes.append('I')
        if polar:
            stokes.extend(['Q', 'U'])

        indices = [['I', 'Q', 'U'].index(s) for s in stokes]

        return stokes, indices

    def _getspec(self, polar=True, temp=False, EBTB=False):
        """
        Get the spectra number and name(s)

        Parameters
        ----------
        polar : boolean, optional
            Append EE, BB spectra to the list of spectra. Default: True
        temp : boolean, optional
            Append TT spectrum to the list of spectra.
            Default is False.
        EBTB : boolean, optional
            Append cross-spectra (according to the value of polar and temp)
            to the list of spectra. Default: False
        """
        allspec = ['TT', 'EE', 'BB', 'TE', 'EB', 'TB']
        der = []
        if temp:
            der.append('TT')

        if polar:
            der.extend(['EE', 'BB'])
            if temp:
                der.append('TE')

            if EBTB:
                if temp:
                    der.extend(['TE', 'EB', 'TB'])
                else:
                    der.append('EB')

        return der, [allspec.index(c) for c in der]

    def _CorrelationMatrix(self, clth):
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
        model = clth[self.ispec][:, 2:int(self.ellbins[-1])].flatten()

        # # Return scalar product btw Pl and the fiducial spectra.
        return np.sum(self.Pl * model[:, None, None], 0)
