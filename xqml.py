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
            self, mask, bins, clth, lmax=None, Pl=None,
            fwhm=0., polar=True, temp=False, EBTB=False):
        """
        Parameters
        ----------
        mask : 1D array of booleans
            Mask defining the region of interest (of value True)
        bins : 1D array of floats
            Bin centers or bin edges?
        clth : nD array of floats
            Array containing fiducial CMB spectra (unbinned).
        lmax : int
            Maximum multipole
        Pl : ??? or None, optional
            ???. Default is None.
        fwhm : float, optional
            FWHM of the experiment beam
        polar : Boolean, optional
            Compute the polarisation part (E and B). Default is True.
        temp : Boolean, optional
            Compute the temperature part (T). Default is False.
        EBTB : Boolean, optional
            ???. Default is False.

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
        self.spec, self.ispec = self._getspec(polar=polar, temp=temp, EBTB=EBTB)
        self.nstokes = len(self.stokes)
        self.nspec = len(self.spec)

        # If Pl is given by the user, just load it, and then compute the signal
        # covariance using the fiducial model.
        # Otherwise compute Pl and S from the arguments.
        if Pl is None:
            self.Pl, self.S = compute_ds_dcb(
                self.ellbins, self.nside, self.ipok,
                self.bl, clth, self.Slmax,
                polar=polar, temp=temp, EBTB=EBTB,
                pixwining=True, timing=True, MC=False)
        else:
            self.Pl = Pl
            self.S = self._CorrelationMatrix(clth)

        # Reshape Pl into a square matrix (2D array)
        self.Pl = self.Pl.reshape(
            -1, self.nstokes * self.npix, self.nstokes * self.npix)

    def construct_esti(self, NA, NB):
        """
        ???

        Parameters
        ----------
        NA : 2D array
            ???
        NB : 2D array
            ???

        """
        # Invert (signalA + noise) matrix
        self.invCa = linalg.inv(self.S + NA)

        # Invert (signalB + noise) matrix
        self.invCb = linalg.inv(self.S + NB)

        # Compute E using Eq...
        self.E = El( self.invCa, self.invCb, self.Pl)

        # Finally compute invW by inverting...
        self.invW = linalg.inv(CrossWindowFunction(self.E, self.Pl))

    def get_spectra(self, map1, map2):
        """
        ???

        Parameters
        ----------
        NA : 2D array
            ???
        NB : 2D array
            ???

        """
        # Define conditions based on the map size
        cond_size1 = np.size(map1) == self.nstokes * self.npix
        cond_size2 = np.size(map2) == self.nstokes * self.npix

        d1 = map1 if cond_size1 else map1[self.istokes,self.mask]
        d2 = map2 if cond_size2 else map2[self.istokes,self.mask]

        # yl is...
        yl = yQuadEstimator(d1.ravel(), d2.ravel(), self.E)

        # cl is obtained using...
        cl = ClQuadEstimator(self.invW, yl)

        # Return the reshaped set of cls
        return cl.reshape(self.nspec,-1)

    def get_covariance(self):
        """
        ???

        Returns
        ----------
        V : ???
            V is ....

        """
        ## Do...
        G = CrossGisherMatrix( self.E, self.S)

        ## Compute V using....
        V = CovAB(self.invW, G)

        return(V)

    def _getstokes(self, polar=True, temp=False):
        """
        ???

        Parameters
        ----------
        polar : Boolean, optional
            Append Q, U to the list of Stokes parameters. Default is True.
        temp : Boolean, optional
            Append T to the list of Stokes parameters. Default is False.

        Returns
        ----------
        stokes : List of strings
            List containing I or/and Q, U.
        indices : List of strings
            List containing position of I, Q and U in the
            stokes list (if present).
        """
        stokes=[]
        if temp:
            stokes.append( 'I')
        if polar:
            stokes.extend( ['Q','U'])

        indices = [['I','Q','U'].index(s) for s in stokes]

        return stokes, indices

    def _getspec(self, polar=True, temp=False, EBTB=False):
        """
        ???

        Parameters
        ----------
        polar : Boolean, optional
            Append EE, BB spectra to the list of spectra. Default is True.
        temp : Boolean, optional
            Append TT spectrum to the list of spectra.
            Default is False.
        EBTB : Boolean, optional
            Append cross-spectra (according to the value of polar and temp)
            to the list of spectra. Default is False.
        """
        allspec = ['TT','EE','BB','TE','EB','TB']
        der = []
        if temp:
            der.append( 'TT')

        if polar:
            der.extend( ['EE','BB'])
            if temp:
                der.append( 'TE')

            if EBTB:
                if temp:
                    der.extend(['TE','EB','TB'])
                else:
                    der.append( 'EB')

        return der, [allspec.index(c) for c in der]

    def _CorrelationMatrix(self, clth):
        """
        Compute correlation matrix S = sum_l Pl*Cl

        QUESTION: why you need to pass clth while it is in the constructor of
        the class already? you could just register it as
        an attribute (self.clth) and call it here.

        Parameters
        ----------
        clth : nD array of floats
            Array containing fiducial CMB spectra (unbinned).
        """
        # Choose only needed spectra according to ispec, and truncate
        # the ell range according the bin range. Flatten (1D) the result.
        model = clth[self.ispec][:,2:int(self.ellbins[-1])].flatten()

        ## Return scalar product btw Pl and the fiducial spectra.
        return np.sum(self.Pl*model[:,None, None], 0)
