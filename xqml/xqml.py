"""
(Cross-) power spectra estimation using the QML method.
[Vanneste et al. 2018, arXiv:1807.02484]
"""
from __future__ import division


from time import perf_counter
from scipy import linalg

import numpy as np
import healpy as hp
import random as rd

from .bins import Bins

from . import xqml_utils as xut
from .estimators import El
from .estimators import CovAB
from .estimators import CrossWindowFunction
from .estimators import yQuadEstimator, ClQuadEstimator
from .estimators import biasQuadEstimator

from .libcov import compute_ds_dcb, SignalCovMatrix
from . import _libcov as clibcov
from ._libcov import py_set_threads as set_threads

__all__ = ['xQML', 'Bins', 'set_threads']


class xQML(object):
    """ Main class to handle the spectrum estimation """
    def __init__(self, maskA, maskB, bins, clth,
                 NA=None, NB=None,
                 FA=None, FAmask=None,
                 FB=None, FBmask=None,
                 lmax=None, Pl=None,
                 fwhm=0., bell=None, spec=['EE','BB'], pixwin=True, verbose=True,
                 nthreads=-1):
        """
        Parameters
        ----------
        maskA,maskB : npt.NDArray
            Mask defining the region of interest (of value True)
        bins : Bins class object
            Contains information about bins
        clth : ndarray of floats
            Array containing fiducial CMB spectra (unbinned)
        NA,NB: npt.NDArray=None
        FA,FB: npt.NDArray = None
            Matrix representation of the filter
        FAmask,FBmask: npt.NDArray = None
            Binary mask definine the filter's domain. shape (12*nside^2, )
        lmax : int
            Maximum multipole
        Pl : ndarray or None, optional
            Normalize Legendre polynomials dS/dCl. Default: None
        fwhm : float, optional
            FWHM of the experiment beam in degree
        bell : ndarray, optional
            beam transfer function (priority over fwhm)
        pixwin : bool, optional
            If True, applies pixel window function to spectra. Default: True
        """
        if nthreads>0:
            set_threads(nthreads)
        self.bias = None
        self.cross = NB is not None
        self.NA = NA
        self.NB = NB if self.cross else NA
        # Number of pixels in the mask
        # For example that would be good to have an assertion
        # on the mask size, just to check that it corresponds to a valid nside.
        assert maskA.shape==maskB.shape, "xQML only supports maps at the same resolution"
        # Map resolution (healpix)
        self.nside = hp.get_nside(maskA)
        npixtot = hp.nside2npix(self.nside)
        # ipok are pixel indexes outside the mask
        if FAmask is not None or FBmask is not None:
            _maskA = np.asarray(FAmask if FAmask is not None else maskA, bool)
            _maskB = np.asarray(FBmask if FBmask is not None else maskB, bool)
            if FAmask is not None:
                assert np.all(FAmask[maskA]), "Filter A does not fully cover the mask region A"
            if FBmask is not None:
                assert np.all(FBmask[maskB]), "Filter A does not fully cover the mask region A"
        else:
            _maskA = np.asarray(maskA, bool)
            _maskB = np.asarray(maskB, bool)
        
        # build initial Pl for the extended region
        mask_ext = np.logical_or(_maskA, _maskB)
        ipok_ext = np.arange(npixtot)[mask_ext]
        npix_ext = np.count_nonzero(mask_ext)
        
        # binning (Bins class)
        self.bins = bins
        
        # Maximum multipole based on nside (rule of thumb to avoid aliasing)
        self.Slmax = bins.lmax if lmax is None else lmax
        
        # Beam 2pt function (Gaussian)
        self.bl = hp.gauss_beam(np.deg2rad(fwhm), lmax=self.Slmax)
        if bell is not None:
            self.bl = bell[:self.Slmax+1]
        
        # Set the Stokes parameters needed        
        self.stokes, self.spec, self.istokes, self.ispecs = xut.getstokes(spec)
        self.nstokes = len(self.stokes)
        self.nspec = len(self.spec)
        self.pixwin = pixwin
        
        clth = np.asarray(clth)
        if len(clth) == 4:
            clth = np.concatenate((clth,clth[0:2]*0.))

        if verbose:
            nbin = bins.nbins
            nmem = self.nspec * nbin * (self.nstokes * npix_ext)**2
            toGb = 1024. * 1024. * 1024.
            print("xQML")
            print(f"spec: {spec}")
            print(f"nbin: {nbin}")
            print(f"Memset: {8.*nmem/toGb:.2f} Gb {self.nspec} spec {nbin} bin {self.nstokes} {npix_ext} pix")
        # If Pl is given by the user, just load it, and then compute the signal
        # covariance using the fiducial model.
        # Otherwise, compute Pl and S from the arguments.
        # Ok, but Pl cannot be binned, otherwise S construction is not valid
        tic = perf_counter()
        if Pl is None:
            Pl = compute_ds_dcb(self.bins, self.nside, ipok_ext, self.bl, clth, self.Slmax,
                                     self.spec, pixwin=self.pixwin)
            if verbose:
                print(f"Construct Pl: {perf_counter() - tic:.1f} sec")
            tic = perf_counter()

        else:
            pass

        # restore the total mask
        self.maskA = np.asarray(maskA, bool)
        self.maskB = np.asarray(maskB, bool)
        
        self.ipokA = np.arange(npixtot)[self.maskA]
        self.npixA = np.count_nonzero(self.maskA)
        self.ipokB = np.arange(npixtot)[self.maskB]
        self.npixB = np.count_nonzero(self.maskB)
        # Pl has the extended shape, and is symmetric
        # self.Pl can be rectangular, and has masking/filter applied.
        if FA is not None or FB is not None or np.any(maskA != maskB):
            self.MF_A = self.get_MF(self.maskA, FA, FAmask, mask_ext)
            self.MF_B = self.get_MF(self.maskB, FB, FBmask, mask_ext)
            self.Pl = clibcov.FilterPl(Pl, self.MF_A, self.MF_B)
            if verbose:
                print(f"Filter Pl: {perf_counter() - tic:.1f} sec")
            tic = perf_counter()
        else:
            self.Pl = Pl
        if np.all(clth==0):
            self.SA = np.zeros((self.nstokes*self.npixA, self.nstokes*self.npixA), dtype=np.float64)
            self.SB = np.zeros((self.nstokes * self.npixB, self.nstokes*self.npixB), dtype=np.float64)
            self.SAB = np.zeros((self.nstokes*self.npixA, self.nstokes*self.npixB), dtype=np.float64)
        else:
            P, Q = bins._bin_operators()
            S = SignalCovMatrix(self.Pl, np.array([P.dot(clth[isp, :bins.lmax + 1]) for isp in
                                                   self.ispecs]).ravel())
            self.SAB = S
            
            if not hasattr(self, 'MF_A') and not hasattr(self, 'MF_B'):
                self.SA = self.SB = S
            elif np.all(self.MF_A==self.MF_B):
                self.SA = self.SB = S
            else:
                PlA = clibcov.FilterPl(Pl, self.MF_A, self.MF_A)
                PlB = clibcov.FilterPl(Pl, self.MF_B, self.MF_B)
                if verbose:
                    print(f"Filter Pl separately for A and B: {perf_counter() - tic:.1f} sec")
                tic = perf_counter()
                self.SA = SignalCovMatrix(PlA, np.array([P.dot(clth[isp, :bins.lmax + 1]) for isp in
                                                         self.ispecs]).ravel())
                self.SB = SignalCovMatrix(PlB, np.array([P.dot(clth[isp, :bins.lmax + 1]) for isp in
                                                         self.ispecs]).ravel())
        del Pl
        if verbose:
            print(f"Construct S: {perf_counter() - tic:.1f} sec")
        tic = perf_counter()
        
        if NA is not None:
            self.construct_esti(NA=NA, NB=NB, verbose=verbose, thread=True)
        if verbose:
            print(f"Construct estimator: {perf_counter()-tic:.1f} sec")
    
    def get_MF(self, mask, F, Fmask, mask_ext):
        """
        compute the masking-filter matrix, shape(npix, npix_ext)
        Parameters
        ----------
        F
        Fmask
        mask
        mask_ext
        Returns
        -------

        """
        def maskM(mask1, mask2):
            """
            mask1 is a subset of mask2
            Parameters
            ----------
            mask1
            mask2

            Returns
            -------
            np.ndarray shape (nstokes*npix1, nstoks*npix2)
            """
            npix1 = np.count_nonzero(mask1)
            npix2 = np.count_nonzero(mask2)
            out = np.zeros((self.nstokes*npix1, self.nstokes*npix2), dtype=np.float64)
            ipix = np.arange(hp.nside2npix(self.nside))
            pix_map = np.isin(ipix[mask2], ipix[mask1], )
            for i in range(self.nstokes):
                s_ = np.s_[np.arange(npix1) + i * npix1, np.arange(npix2)[pix_map] + i * npix2]
                out[s_] = 1
            return out
        
        if F is None:
            return maskM(mask, mask_ext)
        else:
            M = maskM(Fmask, mask_ext) # npix_full X npix_ext
            npix = np.count_nonzero(mask)
            ipix = np.arange(hp.nside2npix(self.nside))
            pix_map = np.isin(ipix[Fmask], ipix[mask])
            nF = np.count_nonzero(Fmask)
            assert self.nstokes * nF==F.shape[0]
            MF = np.zeros((self.nstokes * npix, F.shape[1]))
            for i in range(self.nstokes):
                MF[i * npix: (i + 1) * npix] = F[i * nF:(i + 1) * nF, :][pix_map]
            MF = np.dot(MF, M)
            return MF
    
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
        verbose: bool=False
        thread: bool=True
            do OMP or threading
        """
        
        # Invert (signalA + noise) matrix
        invCa = xut.pd_inv(self.SA + NA)

        # Invert (signalB + noise) matrix
        invCb = xut.pd_inv(self.SB + NB)
        
        # Compute El = Ca^-1.Pl.Cb^-1 (long)
        # Compute El = Cb^-1.Pl^T.Ca^-1 (long)
        self.El = El(invCa, invCb, self.Pl, verbose=verbose)
        
        # Finally compute invW by inverting (longer)
        self.invW = linalg.inv(CrossWindowFunction(self.El, self.Pl, verbose=verbose))
        
        # Compute bias for auto
        if not self.cross:
            self.bias = biasQuadEstimator(NA, self.El)
            
    def get_spectra(self, mapA, mapB=None):
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
        cond_sizeA = np.size(mapA)==self.nstokes * self.npixA
        dA = mapA if cond_sizeA else mapA[self.istokes][:, self.maskA]

        if self.cross:
            cond_sizeB = np.size(mapB)==self.nstokes * self.npixB
            dB = mapB if cond_sizeB else mapB[self.istokes][:, self.maskB]
            yl = clibcov.yQuadEstimator(dA.ravel(), dB.ravel(), self.El)
        else:
            yl = clibcov.yQuadEstimator(dA.ravel(), dA.ravel(), self.El) - self.bias

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
        if self.cross:
            G = clibcov.CrossGisher(self.SAB, self.El)
        else:
            # G = CrossGisherMatrix(self.El, self.S + self.NA)
            G = clibcov.CrossGisher(self.SAB+self.NA, self.El)

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
        P, Q = self.bins._bin_operators()
        Cl = xut.Cl4to6(clth)
        S = SignalCovMatrix(self.Pl, np.array([P.dot(Cl[isp, :self.bins.lmax + 1]) for isp in self.ispecs]).ravel())
        return S

    def __call__(self, mapA, mapB=None, return_cov=False):
        """
        Return the unbiased spectra
        Parameters
        ----------
        mapA, mapB : 1D array
            Pixel map number 1/2. The maps should have shape (3, npix) or (nstoeks*npix_masked), in the former case, the
            masking will be applied to the maps.
        return_cov: bool=True

        Returns
        ----------
        Cl: np.ndarray
            Returns cl or a list of cl's (in order of TT, EE, BB, TE, TB, EB)
        [Cl_cov, ]: np.ndarray
            2D ell-ell covariance matrix. Note that computing this is very heavy and double the memory footprint!
        """
        Cl = self.get_spectra(mapA, mapB)
        if not return_cov:
            return Cl
        cov = self.get_covariance()
        return Cl, cov