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
            fwhm=0., spec=None, temp=False, polar=True, corr=False,
            pixwin=True):
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
        corr : boolean, optional
            If True, compute TE, TB, EB spectra. Default: False
        pixwin : boolean, optional
            If True, applies pixel window function to spectra. Default: True

        """
        # Number of pixels in the mask
        # For example that would be good to have an assertion
        # on the mask size, just to check that it corresponds to a valid nside.
        npixtot = len(mask)

        # Map resolution (healpix)
        self.nside = hp.npix2nside(npixtot)

        # ipok are pixel indexes outside the mask
        self.ipok = np.arange(npixtot)[np.array(mask, bool)]
        self.mask = mask
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

        self.stokes, self.spec, self.istokes, self.ispecs = self._getstokes(
            spec, temp, polar, corr)
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
                self.spec, pixwin=pixwin, timing=True, MC=False)
        else:
            self.Pl = Pl
            if S is None:
                self.S = self._CorrelationMatrix(clth)
            else:
                self.S = S
        if S is not None:
            self.S = S

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

    def _getstokes(self, spec=None, temp=False, polar=False, corr=False):
        """
        Get the Stokes parameters number and name(s)

        Parameters
        ----------
        spec : bool
            If True, get Stokes parameters for polar (default: True)
        polar : bool
            If True, get Stokes parameters for polar (default: True)
        temp : bool
            If True, get Stokes parameters for temperature (default: False)
        corr : bool
            If True, get Stokes parameters for EB and TB (default: False)

        Returns
        ----------
        stokes : list of string
            Stokes variables names
        spec : int
            Spectra names
        istokes : list
            Indexes of power spectra

        Example
        ----------
        >>> getstokes(polar=True, temp=False, corr=False)
        (['Q', 'U'], ['EE', 'BB'], [1, 2])
        >>> getstokes(polar=True, temp=True, corr=False)
        (['I', 'Q', 'U'], ['TT', 'EE', 'BB', 'TE'], [0, 1, 2, 3])
        >>> getstokes(polar=True, temp=True, corr=True)
        (['I', 'Q', 'U'], ['TT', 'EE', 'BB', 'TE', 'EB', 'TB'], [0, 1, 2, 3, 4,
        5])
        """
        if spec is not None:
            _temp = "TT" in spec or "TE" in spec or "TB" in spec or temp
            _polar = "EE" in spec or "BB" in spec or "TE" in spec or "TB" in \
                spec or "EB" in spec or polar
            _corr = "TE" in spec or "TB" in spec or "EB" in spec or corr
        else:
            _temp = temp
            _polar = polar
            _corr = corr

        speclist = []
        if _temp or (spec is None and corr):
            speclist.extend(["TT"])
        if _polar:
            speclist.extend(["EE", "BB"])
        if spec is not None and not corr:
            if 'TE' in spec:
                speclist.extend(["TE"])
            if 'EB' in spec:
                speclist.extend(["EB"])
            if 'TB' in spec:
                speclist.extend(["TB"])

        elif _corr:
            speclist.extend(["TE", "EB", "TB"])

        stokes = []
        if _temp:
            stokes.extend(["I"])
        if _polar:
            stokes.extend(["Q", "U"])

        ispecs = [['TT', 'EE', 'BB', 'TE', 'EB', 'TB'].index(s) for s
                  in speclist]
        istokes = [['I', 'Q', 'U'].index(s) for s in stokes]
        return stokes, speclist, istokes, ispecs

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
        model = clth[self.ispecs][:, 2:int(self.ellbins[-1])].flatten()

        # # Return scalar product btw Pl and the fiducial spectra.
        return np.sum(self.Pl * model[:, None, None], 0)
