"""
(Cross-) power spectra estimation using the QML method.
[Vanneste et al. 2018, arXiv:1807.02484]
"""
from __future__ import division

import sys
import math
import timeit
import string

from scipy import special, linalg, sparse

import numpy as np
import healpy as hp
import random as rd

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


















#################################  Spin functions  ###############################################
def dlss(z, s1, s2, lmax):
  '''
  Matt version
  '''
  d = np.zeros((lmax+1))
  if s1 < abs(s2):
    print("error spins, s1<|s2|")
    return
  # sign = -1 if (s1 + s2) and 1 else 1
  sign = (-1)**(s1-s2)
  d[s1] = sign / 2.**s1 * math.sqrt(math.factorial(2.*s1)/math.factorial(1.*s1+s2)/math.factorial(1.*s1-s2)) *  (1.+z)**(.5*(s1+s2)) *  (1.-z)**(.5*(s1-s2))

  l1 = s1+1.
  rhoSSL1 = math.sqrt( (l1*l1 - s1*s1) * (l1*l1 - s2*s2) ) / l1
  d[s1+1] = (2*s1+1.)*(z-s2/(s1+1.)) * d[s1] / rhoSSL1
  for l in np.arange(s1+1, lmax, 1) : #( l=s1+1; l<lmax; l++) {
    l1 = l+1.
    rhoSSL  = math.sqrt( (l*l*1.- s1*s1) * (l*l*1.- s2*s2) ) / (l*1.)
    rhoSSL1 = math.sqrt( (l1*l1 - s1*s1) * (l1*l1 - s2*s2) ) / l1
    d[l+1] = ( (2.*l+1.)*(z-s1*s2/(l*1.)/l1)*d[l] - rhoSSL*d[l-1] ) / rhoSSL1
  return d


def pl0(z,lmax):
  '''
  Compute sequence of Legendre functions of the first kind (polynomials),
    Pn(z) and derivatives for all degrees from 0 to lmax (inclusive).
    '''
  return special.lpn(lmax,z)[0] # 0 is for no derivative?

def pl2(z,lmax):
  '''
  Computes the associated Legendre function of the first kind of order m and
    degree n, ``Pmn(z)`` = :math:`P_n^m(z)`, and its derivative
    '''
  return(special.lpmn(2,lmax,z)[0][2])


######## F1 and F2 functions from Tegmark & De Oliveira-Costa, 2000  ################
def F1l0(z,lmax):
  if abs(z)==1.0:
    return(np.zeros(lmax+1))
  else:
    ell=np.arange(lmax+1)
    thepl=pl0(z,lmax)
    theplm1=np.append(0,pl0(z,lmax-1))
    a0=2./np.sqrt((ell-1)*ell*(ell+1)*(ell+2))
    a1=ell*z*theplm1/(1-z**2)
    a2=(ell/(1-z**2)+ell*(ell-1)/2)*thepl
    bla=a0*(a1-a2)
    bla[0]=0. # deux premiers poles
    bla[1]=0. # deux premiers poles
    return bla

def F1l2(z,lmax):
  if z==1.0:
    # return(np.ones(lmax+1)*0.5) # former version
    return np.append(np.zeros(2),np.ones(lmax-1)*0.5) # = 0 pour l=0,1
  elif z==-1.0:
    ell=np.arange(lmax+1)
    # return(0.5*(-1)**ell) # former version
    return np.append(np.zeros(2),0.5*(-1)**ell[2:]) # = 0 pour l=0,1
  else:
    ell=np.arange(lmax+1)
    thepl2=pl2(z,lmax)
    theplm1_2=np.append(0,pl2(z,lmax-1))
    a0=2./((ell-1)*ell*(ell+1)*(ell+2))
    a1=(ell+2)*z*theplm1_2/(1-z**2)
    a2=((ell-4)/(1-z**2)+ell*(ell-1)/2)*thepl2
    bla=a0*(a1-a2)
    bla[0]=0. # deux premiers poles
    bla[1]=0. # deux premiers poles
    return bla

def F2l2(z,lmax):
  if z==1.0:
    # return(-0.5*np.ones(lmax+1)) # former version
    return np.append(np.zeros(2),-0.5*np.ones(lmax-1)) # = 0 pour l=0,1
  elif z==-1.0:
    ell=np.arange(lmax+1)
    # return(0.5*(-1)**ell) # former version
    return np.append(np.zeros(2),0.5*(-1)**ell[2:]) # syl : = 0 pour l=0,1
  else:
    ell=np.arange(lmax+1)
    thepl2=pl2(z,lmax)
    theplm1_2=np.append(0,pl2(z,lmax-1))
    a0=4./((ell-1)*ell*(ell+1)*(ell+2)*(1-z**2))
    a1=(ell+2)*theplm1_2
    a2=(ell-1)*z*thepl2
    bla=a0*(a1-a2)
    bla[0]=0 # ??
    bla[1]=0 # ??
    return bla



################################# Rotation Angles ###############################################
def polrotangle(ri,rj):
    z=np.array([0.,0.,1.])

    # Compute ri^rj : unit vector for the great circle connecting i and j
    rij=np.cross(ri,rj)
    norm=np.sqrt(np.dot(rij,np.transpose(rij)))

    # case where pixels are identical or diametrically opposed on the sky
    if norm<=1e-15:
        cos2a=1.
        sin2a=0.
        return cos2a,sin2a
    rij=rij/norm

    # Compute z^ri : unit vector for the meridian passing through pixel i
    ris=np.cross(z,ri)
    norm=np.sqrt(np.dot(ris,np.transpose(ris)))

    # case where pixels is at the pole
    if norm<=1e-15:
        cos2a=1.
        sin2a=0.
        return cos2a,sin2a
    ris=ris/norm

    # Now, the angle we want is that between these two great circles: defined by
    cosa=np.dot(rij,np.transpose(ris))

    # the sign is more subtle : see tegmark et de oliveira costa 2000 eq. A6
    rijris=np.cross(rij,ris)
    sina=np.dot(rijris,np.transpose(ri))

    # so now we have directly cos2a and sin2a
    cos2a=2.*cosa*cosa-1.
    sin2a=2.*cosa*sina
    return cos2a,sin2a








############################################## ds_dcb ##################################################
# compute_ds_dcb calls either covth_bins_fast or covth_bins_fast_compressed

def compute_ds_dcb(ellbins,nside,ipok,bl, clth, Slmax, polar=True,temp=True,EBTB=False, pixwining=False, timing=False, MC=0, Sonly=False):
  print('dS/dCb Calulation:')
  print('Temp='+str(temp))
  print('Polar='+str(polar))
  print('EBTB='+str(EBTB))
  print('pixwining='+str(pixwining))
  # print('compressed='+str(compressed))

  if Slmax < ellbins[-1]:
    "WARNING : Slmax < lmax"

  # Slmax+=1 # for max binning value
  allStoke,der,ind = getstokes(polar=polar,temp=temp,EBTB=EBTB)
  print('Stokes parameters :',allStoke)
  print('Derivatives w.r.t. :',der)

  nder=len(der)

  #### define pixels
  rpix=np.array(hp.pix2vec(nside,ipok))
  allcosang=np.dot(np.transpose(rpix),rpix)
  allcosang[allcosang>1]=1.
  allcosang[allcosang<-1]=-1.

  ### dimensions of the large matrix and loop for filling it
  nstokes=len(allStoke)
  nbins=len(ellbins)-1
  npix=len(ipok)

  start = timeit.default_timer()
  if Sonly:
    if MC:
      Smatrix = S_bins_MC(ellbins,nside,ipok,allcosang,bl,clth,Slmax,MC,polar=polar,temp=temp, EBTB=EBTB, pixwining=pixwining, timing=timing)
    else:
      Smatrix = S_bins_fast(ellbins,nside,ipok,allcosang,bl,clth,Slmax,polar=polar,temp=temp, EBTB=EBTB, pixwining=pixwining, timing=timing)
    return Smatrix

  if MC:
    dcov, Smatrix = covth_bins_MC(ellbins,nside,ipok,allcosang,bl,clth,Slmax,MC,polar=polar,temp=temp, EBTB=EBTB, pixwining=pixwining, timing=timing)
  else:
    dcov, Smatrix = covth_bins_fast(ellbins,nside,ipok,allcosang,bl,clth,Slmax,polar=polar,temp=temp, EBTB=EBTB, pixwining=pixwining, timing=timing)

  stop = timeit.default_timer()
  print("Time of computing : " + str(stop - start))

  return (dcov, Smatrix)




def covth_bins_MC(ellbins,nside,ipok,allcosang,bl,clth, Slmax, nsimu, polar=True,temp=True,EBTB=False, pixwining=False, timing=False):
  '''
  Can be particularly slow on sl7 !
  '''
  if nsimu==1:
    nsimu=(12*nside**2)*10*(int(polar)+1)
  print("nsimu=", nsimu)
  lmax=ellbins[-1]
  ell=np.arange(np.min(ellbins),np.max(ellbins)+1)
  maskl=ell<(lmax+1) # assure que ell.max < lmax
  ell=ell[maskl]
  nbins=len(ellbins)-1
  minell=np.array(ellbins[0:nbins]) # define min
  maxell=np.array(ellbins[1:nbins+1])-1 # and max of a bin
  ellval=(minell+maxell)*0.5

  print('minell:',minell)
  print('maxell:',maxell)

  #### define Stokes
  allStoke,der,ind = getstokes(polar=polar,temp=temp,EBTB=EBTB)
  nder=len(der)

  #### define pixels
  rpix=np.array(hp.pix2vec(nside,ipok))
  maskl=ell<(lmax+1) # assure que ell.max < lmax

  #### define Pixel window function
  ll = np.arange(Slmax+2) # +1 avant Slmax+=1

  if pixwining:
    prepixwin=np.array(hp.pixwin(nside, pol=True))
    poly = np.polyfit(np.arange(len(prepixwin[1,2:])), np.log(prepixwin[1,2:]), deg=3, w=np.sqrt(prepixwin[1,2:]))
    y_int  = np.polyval(poly, np.arange(Slmax)) # -1 avant Slmax+=1
    fpixwin = np.append([0,0], np.exp(y_int))
    # fpixwin = np.array(hp.pixwin(nside, pol=True))[int(polar)][2:]
  else:
    fpixwin=ll*0+1

  print("shape fpixwin", np.shape(fpixwin))
  print("shape bl", np.shape(bl[:Slmax+2])) # +1 avant Slmax+=1
  masks=[]
  for i in np.arange(nbins):
      masks.append((ll[:]>=minell[i]) & (ll[:]<=maxell[i]))
  masks = np.array(masks)
  print('Bins mask shape :', np.shape(masks))
  print(masks*1)

  npix = len(ipok)
  start = timeit.default_timer()
  norm = bl[0:Slmax+2]**2*fpixwin[0:Slmax+2]**2 # +1 avant Slmax+=1

  if polar:
    # stok = [1,2,4] if EBTB else [1,2]
    # npix=2*npix
    ClthOne = np.zeros((nder*(nbins), 6,(Slmax+2))) # avant Slmax+=1
    for l in np.arange(2*nbins):
      ClthOne[l,l/nbins+1]= masks[l % nbins]*norm
    if EBTB:
      print("not implemented")
      # break;
      for l in np.arange(2*nbins, 3*nbins):
        ClthOne[l,1]= masks[l % nbins]*norm
        ClthOne[l,2]= masks[l % nbins]*norm
        ClthOne[l,4]= masks[l % nbins]*norm #-nbins*(l/nbins)]*norm # couille ici
    dcov = np.zeros((nder*(nbins), 2*npix, 2*npix))
    start = timeit.default_timer()
    for l in np.arange((nder*nbins)):
      progress_bar(l,nder*(nbins), -(start-timeit.default_timer()))
      dcov[l] = np.cov(np.array([np.array(hp.synfast(ClthOne[l], nside, lmax=Slmax, new=True,verbose=False))[1:3,ipok].flatten() for s in np.arange(nsimu)]).reshape(nsimu, 2*npix), rowvar=False)

    dcov = dcov.reshape(nder,nbins,2*npix,2*npix)
    S = np.cov(np.array([np.array(hp.synfast(clth[:,:Slmax+2]*norm, nside, lmax=Slmax, new=True,verbose=False))[1:3,ipok].flatten() for s in np.arange(nsimu)]).reshape(nsimu, 2*npix), rowvar=False)

  else:
    ClthOne = np.zeros((nbins, (Slmax+2)))
    for l in np.arange((nbins)):
      ClthOne[l]= masks[l]*norm
    dcov = np.zeros(((nbins), npix, npix))
    for l in np.arange((nbins)):
      progress_bar(l,(lmax+1), -(start-timeit.default_timer()))
      dcov[l] = np.cov(np.array([hp.synfast(ClthOne[l], nside, lmax=Slmax, verbose=False)[ipok] for s in np.arange(nsimu)]).reshape(nsimu, npix), rowvar=False)
    dcov = dcov.reshape(1,nbins,npix, npix)
    S = np.cov(np.array([np.array(hp.synfast(clth[:, :Slmax+2]*norm, nside, lmax=Slmax, new=True,verbose=False))[0,ipok].flatten() for s in np.arange(nsimu)]).reshape(nsimu, npix), rowvar=False)

  stop = timeit.default_timer()
  print("time [sec] = ", round(stop-start, 2)  )

  return (dcov, S)



def S_bins_MC(ellbins,nside,ipok,allcosang,bl,clth, Slmax, nsimu, polar=True,temp=True,EBTB=False, pixwining=False, timing=False):
  '''
  Can be particularly slow on sl7 !
  '''
  if nsimu==1:
    nsimu=(12*nside**2)*10*(int(polar)+1)#10000
  print("nsimu=", nsimu)
  lmax=ellbins[-1] #
  ell=np.arange(np.min(ellbins),np.max(ellbins)+1)
  maskl=ell<(lmax+1) # assure que ell.max < lmax
  ell=ell[maskl]
  nbins=len(ellbins)-1
  minell=np.array(ellbins[0:nbins]) # define min
  maxell=np.array(ellbins[1:nbins+1])-1 # and max of a bin
  ellval=(minell+maxell)*0.5

  print('minell:',minell)
  print('maxell:',maxell)

  #### define Stokes
  allStoke,der,ind = getstokes(polar=polar,temp=temp,EBTB=EBTB)
  nder=len(der)

  #### define pixels
  rpix=np.array(hp.pix2vec(nside,ipok))
  maskl=ell<(lmax+1) # assure que ell.max < lmax

  #### define Pixel window function
  ll = np.arange(Slmax+2) # +1 avant Slmax+=1

  if pixwining:
    prepixwin=np.array(hp.pixwin(nside, pol=True))
    poly = np.polyfit(np.arange(len(prepixwin[1,2:])), np.log(prepixwin[1,2:]), deg=3, w=np.sqrt(prepixwin[1,2:]))
    y_int  = np.polyval(poly, np.arange(Slmax)) # -1 avant Slmax+=1
    fpixwin = np.append([0,0], np.exp(y_int))
    # fpixwin = np.array(hp.pixwin(nside, pol=True))[int(polar)][2:]
  else:
    fpixwin=ll*0+1

  print("shape fpixwin", np.shape(fpixwin))
  print("shape bl", np.shape(bl[:Slmax+2])) # +1 avant Slmax+=1
  masks=[]
  for i in np.arange(nbins):
      masks.append((ll[:]>=minell[i]) & (ll[:]<=maxell[i]))
  masks = np.array(masks)
  print('Bins mask shape :', np.shape(masks))
  print(masks*1)

  npix = len(ipok)
  start = timeit.default_timer()
  norm = bl[0:Slmax+2]**2*fpixwin[0:Slmax+2]**2 # +1 avant Slmax+=1

  if polar:
    # stok = [1,2,4] if EBTB else [1,2]
    # npix=2*npix
    ClthOne = np.zeros((nder*(nbins), 6,(Slmax+2))) # avant Slmax+=1
    for l in np.arange(2*nbins):
      ClthOne[l,l/nbins+1]= masks[l % nbins]*norm
    if EBTB:
      print("not implemented")
      # break;
      for l in np.arange(2*nbins, 3*nbins):
        ClthOne[l,1]= masks[l % nbins]*norm
        ClthOne[l,2]= masks[l % nbins]*norm
        ClthOne[l,4]= masks[l % nbins]*norm #-nbins*(l/nbins)]*norm # couille ici

    S = np.cov(np.array([np.array(hp.synfast(clth[:,:Slmax+2]*norm, nside, lmax=Slmax, new=True,verbose=False))[1:3,ipok].flatten() for s in np.arange(nsimu)]).reshape(nsimu, 2*npix), rowvar=False)

  else:
    ClthOne = np.zeros((nbins, (Slmax+2)))
    for l in np.arange((nbins)):
      ClthOne[l]= masks[l]*norm
    S = np.cov(np.array([np.array(hp.synfast(clth[:, :Slmax+2]*norm, nside, lmax=Slmax, new=True,verbose=False))[0,ipok].flatten() for s in np.arange(nsimu)]).reshape(nsimu, npix), rowvar=False)

  stop = timeit.default_timer()
  print("time [sec] = ", round(stop-start, 2)  )

  return S



def covth_bins_fast(ellbins,nside,ipok,allcosang,bl,clth, Slmax, polar=True, temp=True, EBTB=False, pixwining=False, timing=False):
  '''
  Computes ds_dcb[nspec, nbins, 2*npix, 2*npix] and signal matrix S[2*npix, 2*npix]
  Fast because :
    - Building ds_dcb  directly in the right shape [nspec, nbins, 2*npix, 2*npix]
    - Compute EE and BB parts of ds_dcb at the same time (using their symmetry propreties).
  '''

  #### define bins in ell
  lmax=ellbins[-1] #
  ell=np.arange(np.min(ellbins),np.max(ellbins)+1)
  maskl=ell<(lmax+1) # assure que ell.max < lmax
  ell=ell[maskl]
  nbins=len(ellbins)-1
  minell=np.array(ellbins[0:nbins]) # define min
  maxell=np.array(ellbins[1:nbins+1])-1 # and max of a bin
  ellval=(minell+maxell)*0.5

  print('minell:',minell)
  print('maxell:',maxell)

  #### define Stokes
  allStoke,der,ind = getstokes(polar=polar,temp=temp,EBTB=EBTB)
  nder=len(der)

  #### define pixels
  rpix=np.array(hp.pix2vec(nside,ipok))

  maskl=ell<(lmax+1) # assure que ell.max < lmax
  masklRQ = (np.arange(lmax+1)>=min(ell)) & (np.arange(lmax+1)<(lmax+1))

  #### define Pixel window function
  ll = np.arange(Slmax+2)

  if pixwining:
    prepixwin=np.array(hp.pixwin(nside, pol=True))
    poly = np.polyfit(np.arange(len(prepixwin[int(polar),2:])), np.log(prepixwin[int(polar),2:]), deg=3, w=np.sqrt(prepixwin[int(polar),2:]))
    y_int  = np.polyval(poly, np.arange(Slmax))
    fpixwin = np.exp(y_int)
    fpixwin = np.append(prepixwin[int(polar)][2:],fpixwin[len(prepixwin[0])-2:])[:Slmax]
  else:
    fpixwin=ll[2:]*0+1

  print("shape fpixwin", np.shape(fpixwin))
  print("shape bl", np.shape(bl[:Slmax+2]))
  # print("long pixwin", fpixwin, "short", np.array(hp.pixwin(nside, pol=True))[int(polar)])
  norm=(2*ll[2:]+1)/(4.*np.pi)*(fpixwin**2)*(bl[2:Slmax+2]**2)
  print("norm ", np.shape(norm))
  print("ell ", np.shape(ell))

  #### define masks for ell bins
  masks=[]
  for i in np.arange(nbins):
      masks.append((ll[2:]>=minell[i]) & (ll[2:]<=maxell[i]))
  masks = np.array(masks)
  print('Bins mask shape :', np.shape(masks))
  print("norm", norm)
  print("fpixwin", fpixwin)
  print("maskl", np.array(maskl)*1)
  print(masks*1)

  ### Create array for covariances matrices per bin
  nbpixok=ipok.size
  nstokes=np.size(allStoke)
  print('creating array')
  newcov=np.zeros((nder,nbins,nstokes*nbpixok,nstokes*nbpixok)) # final ds_dcb
  Smatrix = np.zeros((nstokes*nbpixok,nstokes*nbpixok)) # Signal matrix S

  start = timeit.default_timer()
  for i in np.arange(nbpixok):
    if timing:
      progress_bar(i,nbpixok, -.5*(start-timeit.default_timer()))
    for j in np.arange(i,nbpixok):

      if nstokes==1:
        pl=pl0(allcosang[i,j],Slmax+1)[2:]
        elem = np.sum((norm*pl*clth[0,2:Slmax+2])[:-1])
        Smatrix[i,j] = elem
        Smatrix[j,i] = elem
        for b in np.arange(nbins):
          elem = np.sum((norm*pl)[masks[b]])
          newcov[0,b,i,j] = elem
          newcov[0,b,j,i] = elem

      elif nstokes==2:

        cij,sij=polrotangle(rpix[:,i],rpix[:,j])
        cji,sji=polrotangle(rpix[:,j],rpix[:,i])
        cos_chi = allcosang[i,j]

        # Tegmark version
        Q22 =  F1l2(cos_chi,Slmax+1)[2:] #[masklRQ]
        R22 = -F2l2(cos_chi,Slmax+1)[2:] #[masklRQ] # /!\ signe - !

        # Matt version
        # d20  = dlss(cos_chi, 2,  0, Slmax+1)
        #d2p2 = dlss(cos_chi, 2,  2, Slmax+1)
        #d2m2 = dlss(cos_chi, 2, -2, Slmax+1)
        # P02 = -d20
        #Q22 = ( d2p2 + d2m2 )[2:]/2.
        #R22 = ( d2p2 - d2m2 )[2:]/2.

        elem1  = np.sum( (norm * ( cij*cji*Q22 + sij*sji*R22)*(clth[1,2:Slmax+2]) )[:-1]) # EE on QQ [masklRQ]
        elem2  = np.sum( (norm * (-cij*sji*Q22 + sij*cji*R22)*(clth[1,2:Slmax+2]) )[:-1]) # EE on QU [masklRQ]
        elem3  = np.sum( (norm * ( sij*sji*Q22 + cij*cji*R22)*(clth[1,2:Slmax+2]) )[:-1]) # EE on UU [masklRQ]
        elem4  = np.sum( (norm * (-sij*cji*Q22 + cij*sji*R22)*(clth[1,2:Slmax+2]) )[:-1]) # EE on QU [masklRQ]

        elem3 += np.sum( (norm * ( cij*cji*Q22 + sij*sji*R22)*(clth[2,2:Slmax+2]) )[:-1]) # BB on QQ [masklRQ]
        elem4 -= np.sum( (norm * (-cij*sji*Q22 + sij*cji*R22)*(clth[2,2:Slmax+2]) )[:-1]) # BB on QU [masklRQ]
        elem1 += np.sum( (norm * ( sij*sji*Q22 + cij*cji*R22)*(clth[2,2:Slmax+2]) )[:-1]) # BB on UU [masklRQ]
        elem2 -= np.sum( (norm * (-sij*cji*Q22 + cij*sji*R22)*(clth[2,2:Slmax+2]) )[:-1]) # BB on UQ [masklRQ]

        if EBTB:
          elem = np.sum( (norm * ( Q22 - R22)*(clth[4,2:Slmax+2]) )[:-1] )  #EB on all
          elem1 += ( cji*sij + sji*cij)*elem # EB on QQ
          elem2 += (-sji*sij + cji*cij)*elem # EB on QU
          elem3 += (-sji*cij - cji*sij)*elem # EB on UU
          elem4 += ( cji*cij - sji*sij)*elem # EB on QU

        Smatrix[0*nbpixok+i,0*nbpixok+j] =  elem1 # to 3
        Smatrix[0*nbpixok+i,1*nbpixok+j] =  elem2 # to -4
        Smatrix[1*nbpixok+i,1*nbpixok+j] =  elem3 # to 1
        Smatrix[1*nbpixok+i,0*nbpixok+j] =  elem4 # to -2

        Smatrix[0*nbpixok+j,0*nbpixok+i] =  elem1 # to 3
        Smatrix[1*nbpixok+j,0*nbpixok+i] =  elem2 # to -4
        Smatrix[1*nbpixok+j,1*nbpixok+i] =  elem3 # to 1
        Smatrix[0*nbpixok+j,1*nbpixok+i] =  elem4 # to -2

        for b in np.arange(nbins):
          elem1 = np.sum(( norm*( cij*cji*Q22 + sij*sji*R22) )[masks[b]] )  #EE or BB on QQ
          elem2 = np.sum(( norm*(-cij*sji*Q22 + sij*cji*R22) )[masks[b]] )  #EE or BB on QU
          elem3 = np.sum(( norm*( sij*sji*Q22 + cij*cji*R22) )[masks[b]] )  #EE or BB on UU
          elem4 = np.sum(( norm*(-sij*cji*Q22 + cij*sji*R22) )[masks[b]] )  #EE or BB on UQ

          # # EE ij then ji
          newcov[0,b,i,j]                 =  elem1 # to 3 for BB
          newcov[0,b,i,nbpixok+j]         =  elem2 # to -4
          newcov[0,b,nbpixok+i,nbpixok+j] =  elem3 # to 1
          newcov[0,b,nbpixok+i,j]         =  elem4 # to -2

          newcov[0,b,j,i]                 =  elem1 # to 3
          newcov[0,b,nbpixok+j,i]         =  elem2 # to -4
          newcov[0,b,nbpixok+j,nbpixok+i] =  elem3 # to 1
          newcov[0,b,j,nbpixok+i]         =  elem4 # to -2

          # # BB ij then ji
          newcov[1,b,nbpixok+i,nbpixok+j] =  elem1
          newcov[1,b,nbpixok+i,j]         = -elem2
          newcov[1,b,i,j]                 =  elem3
          newcov[1,b,i,nbpixok+j]         = -elem4

          newcov[1,b,nbpixok+j,nbpixok+i] =  elem1
          newcov[1,b,j,nbpixok+i]         = -elem2
          newcov[1,b,j,i]                 =  elem3
          newcov[1,b,nbpixok+j,i]         = -elem4

          # # EB ij then ji
          if EBTB:
            newcov[2,b,i,j]                 = -elem2-elem4 # on QQ
            newcov[2,b,i,nbpixok+j]         =  elem1-elem3 # on QU
            newcov[2,b,nbpixok+i,nbpixok+j] =  elem2+elem4 # on UU
            newcov[2,b,nbpixok+i,j]         =  elem1-elem3 # on UQ

            newcov[2,b,j,i]                 = -elem2-elem4 #
            newcov[2,b,nbpixok+j,i]         =  elem1-elem3 #
            newcov[2,b,nbpixok+j,nbpixok+i] =  elem2+elem4 #
            newcov[2,b,j,nbpixok+i]         =  elem1-elem3 #

            elemQ22 = np.sum(( norm*( Q22 ) )[masks[b]] )  #EB on all
            elemR22 = np.sum(( norm*( - R22) )[masks[b]] )  #EB on all
            newcov[2,b,0*nbpixok+i,0*nbpixok+j] = ( sij*cji*(elemR22+elemQ22) + cij*sji*(elemQ22+elemR22)) # on QQ
            newcov[2,b,0*nbpixok+i,1*nbpixok+j] = (-sij*sji*(elemR22+elemQ22) + cij*cji*(elemQ22+elemR22)) # on QU
            newcov[2,b,1*nbpixok+i,1*nbpixok+j] = (-cij*sji*(elemR22+elemQ22) - sij*cji*(elemQ22+elemR22)) # on UU
            newcov[2,b,1*nbpixok+i,0*nbpixok+j] = ( cij*cji*(elemR22+elemQ22) - sij*sji*(elemQ22+elemR22)) # on UQ
            newcov[2,b,0*nbpixok+j,0*nbpixok+i] = ( sij*cji*(elemR22+elemQ22) + cij*sji*(elemQ22+elemR22)) # to 3
            newcov[2,b,1*nbpixok+j,0*nbpixok+i] = (-sij*sji*(elemR22+elemQ22) + cij*cji*(elemQ22+elemR22)) # to -4
            newcov[2,b,1*nbpixok+j,1*nbpixok+i] = (-cij*sji*(elemR22+elemQ22) - sij*cji*(elemQ22+elemR22)) # to 1
            newcov[2,b,0*nbpixok+j,1*nbpixok+i] = ( cij*cji*(elemR22+elemQ22) - sij*sji*(elemQ22+elemR22)) # to -2

  return (newcov, Smatrix)



def S_bins_fast(ellbins,nside,ipok,allcosang,bl,clth, Slmax, polar=True,temp=True,EBTB=False, pixwining=False, timing=False ):
  '''
  Computes signal matrix S[2*npix, 2*npix]
  Fast because :
    - Compute EE and BB parts of ds_dcb at the same time (using their symmetry propreties).
  '''

  lmax=ellbins[-1] #
  ell=np.arange(np.min(ellbins),np.max(ellbins)+1)
  maskl=ell<(lmax+1) # assure que ell.max < lmax
  ell=ell[maskl]
  nbins=len(ellbins)-1
  minell=np.array(ellbins[0:nbins]) # define min
  maxell=np.array(ellbins[1:nbins+1])-1 # and max of a bin
  ellval=(minell+maxell)*0.5

  print('minell:',minell)
  print('maxell:',maxell)

  #### define Stokes
  allStoke=['I','Q','U']
  if EBTB:
    der=['TT','EE','BB','TE','EB','TB']
    ind=[1,2,3,4,5,6]
  else:
    der=['TT','EE','BB','TE']
    ind=[1,2,3,4]
  if  not temp:
    allStoke=['Q','U']
    if EBTB:
      der=['EE','BB','EB']
      ind=[2,3,5]
    else:
      der=['EE','BB']
      ind=[2,3]
  if not polar:
    allStoke=['I']
    der=['TT']
    ind=[1]

  nder=len(der)

  #### define pixels
  rpix=np.array(hp.pix2vec(nside,ipok))

  maskl=ell<(lmax+1) # assure que ell.max < lmax
  masklRQ = (np.arange(lmax+1)>=min(ell)) & (np.arange(lmax+1)<(lmax+1))

  #### define Pixel window function
  ll = np.arange(Slmax+2)

  if pixwining:
    prepixwin=np.array(hp.pixwin(nside, pol=True))
    poly = np.polyfit(np.arange(len(prepixwin[int(polar),2:])), np.log(prepixwin[int(polar),2:]), deg=3, w=np.sqrt(prepixwin[int(polar),2:]))
    y_int  = np.polyval(poly, np.arange(Slmax))
    fpixwin = np.exp(y_int)
    fpixwin = np.append(prepixwin[int(polar)][2:],fpixwin[len(prepixwin[0])-2:])[:Slmax]
    # fpixwin = np.array(hp.pixwin(nside, pol=True))[int(polar)][2:]
  else:
    fpixwin=ll[2:]*0+1

  print("shape fpixwin", np.shape(fpixwin))
  print("shape bl", np.shape(bl[:Slmax+2]))
  # print("long pixwin", fpixwin, "short", np.array(hp.pixwin(nside, pol=True))[int(polar)])
  norm=(2*ll[2:]+1)/(4.*np.pi)*(fpixwin**2)*(bl[2:Slmax+2]**2)
  print("norm ", np.shape(norm))
  print("ell ", np.shape(ell))

  #### define masks for ell bins
  masks=[]
  for i in np.arange(nbins):
      # masks.append((ell>=minell[i]) & (ell<=maxell[i]))
      masks.append((ll[2:]>=minell[i]) & (ll[2:]<=maxell[i]))
  masks = np.array(masks)
  print('Bins mask shape :', np.shape(masks))
  print("norm", norm)
  print("fpixwin", fpixwin)
  print("maskl", np.array(maskl)*1)
  print(masks*1)

  ### Create array for covariances matrices per bin
  nbpixok=ipok.size
  nstokes=np.size(allStoke)
  print('creating array')
  # newcov=np.zeros((nder,nbins,nstokes*nbpixok,nstokes*nbpixok)) # final ds_dcb
  Smatrix = np.zeros((nstokes*nbpixok,nstokes*nbpixok)) # Signal matrix S

  start = timeit.default_timer()
  for i in np.arange(nbpixok):
    if timing:
      progress_bar(i,nbpixok, -.5*(start-timeit.default_timer()))
    for j in np.arange(i,nbpixok):

      if nstokes==1:
        pl=pl0(allcosang[i,j],Slmax+1)[2:]
        elem = np.sum((norm*pl*clth[0,2:Slmax+2])[:-1])
        Smatrix[i,j] = elem
        Smatrix[j,i] = elem

      elif nstokes==2:

        cij,sij=polrotangle(rpix[:,i],rpix[:,j])
        cji,sji=polrotangle(rpix[:,j],rpix[:,i])
        cos_chi = allcosang[i,j]

        # # # JC version
        # Q22 =  F1l2(cos_chi,Slmax+1)[2:] #[masklRQ]
        # R22 = -F2l2(cos_chi,Slmax+1)[2:] #[masklRQ] # /!\ signe - !

        # # # Matt version
        # d20  = dlss(cos_chi, 2,  0, Slmax+1)
        d2p2 = dlss(cos_chi, 2,  2, Slmax+1)
        d2m2 = dlss(cos_chi, 2, -2, Slmax+1)
        # # # P02 = -d20
        Q22 = ( d2p2 + d2m2 )[2:]/2.
        R22 = ( d2p2 - d2m2 )[2:]/2.

        elem1  = np.sum( (norm * ( cij*cji*Q22 + sij*sji*R22)*(clth[1,2:Slmax+2]) )[:-1]) # EE on QQ [masklRQ]
        elem2  = np.sum( (norm * (-cij*sji*Q22 + sij*cji*R22)*(clth[1,2:Slmax+2]) )[:-1]) # EE on QU [masklRQ]
        elem3  = np.sum( (norm * ( sij*sji*Q22 + cij*cji*R22)*(clth[1,2:Slmax+2]) )[:-1]) # EE on UU [masklRQ]
        elem4  = np.sum( (norm * (-sij*cji*Q22 + cij*sji*R22)*(clth[1,2:Slmax+2]) )[:-1]) # EE on QU [masklRQ]

        elem3 += np.sum( (norm * ( cij*cji*Q22 + sij*sji*R22)*(clth[2,2:Slmax+2]) )[:-1]) # BB on QQ [masklRQ]
        elem4 -= np.sum( (norm * (-cij*sji*Q22 + sij*cji*R22)*(clth[2,2:Slmax+2]) )[:-1]) # BB on QU [masklRQ]
        elem1 += np.sum( (norm * ( sij*sji*Q22 + cij*cji*R22)*(clth[2,2:Slmax+2]) )[:-1]) # BB on UU [masklRQ]
        elem2 -= np.sum( (norm * (-sij*cji*Q22 + cij*sji*R22)*(clth[2,2:Slmax+2]) )[:-1]) # BB on UQ [masklRQ]

        if EBTB:
          elem = np.sum( (norm * ( Q22 - R22)*(clth[4,2:Slmax+2]) )[:-1] )  #EB on all
          elem1 += ( cji*sij + sji*cij)*elem # EB on QQ
          elem2 += (-sji*sij + cji*cij)*elem # EB on QU
          elem3 += (-sji*cij - cji*sij)*elem # EB on UU
          elem4 += ( cji*cij - sji*sij)*elem # EB on QU

        Smatrix[0*nbpixok+i,0*nbpixok+j] =  elem1 # to 3
        Smatrix[0*nbpixok+i,1*nbpixok+j] =  elem2 # to -4
        Smatrix[1*nbpixok+i,1*nbpixok+j] =  elem3 # to 1
        Smatrix[1*nbpixok+i,0*nbpixok+j] =  elem4 # to -2

        Smatrix[0*nbpixok+j,0*nbpixok+i] =  elem1 # to 3
        Smatrix[1*nbpixok+j,0*nbpixok+i] =  elem2 # to -4
        Smatrix[1*nbpixok+j,1*nbpixok+i] =  elem3 # to 1
        Smatrix[0*nbpixok+j,1*nbpixok+i] =  elem4 # to -2

  return (Smatrix)





#################################  Estimator tools #################################

def Pl(ds_dcb):
  '''
  Reshape ds_dcbin Pl
  '''
  nnpix = np.shape(ds_dcb)[-1]
  return np.copy(ds_dcb).reshape(2*(np.shape(ds_dcb)[1]), nnpix, nnpix)

def CorrelationMatrix(Clth, Pl, ellbins, pola=True, temp=False, EBTB=False ):
  '''
  Compute correlation matrix S = sum_l Pl*Cl
  '''
  if EBTB:
    xx=['TT','EE','BB','TE','EB','TB']
    ind=[0,1,2,3,4,5]
  else:
    xx=['TT','EE','BB','TE']
    ind=[0,1,2,3]

  if  not temp:
    allStoke=['Q','U']
    if EBTB:
      xx=['EE','BB','EB']
      ind=[1,2,5]
    else:
      xx=['EE','BB']
      ind=[1,2]
  if not pola:
    allStoke=['I']
    xx=['TT']
    ind=[0]

  clth = Clth[ind][:,2:int(ellbins[-1])].flatten()
  return np.sum(Pl*clth[:,None, None], 0)

def El(invCAA, invCBB, Pl, Bll=None, expend=False):
  '''
  Compute El = invCAA.Pl.invCBB
  '''
  if Bll == None: # Tegmark B-matrix useless so far)
    Bll = np.diagflat((np.ones(len(Pl))))

  lmax = len(Pl)*2**int(expend)
  lrange = np.arange(lmax)
  npix = len(invCAA)
  if expend: # triangular shape ds_dcb
    El = np.array([np.dot(np.dot(invCAA, Expd(Pl,l)), invCBB) for l in lrange]).reshape((lmax,npix,npix ))
  else:
    El = np.array([np.dot(np.dot(invCAA, Pl[l]), invCBB) for l in lrange]).reshape((lmax,npix,npix ))
  return El

def ElLong(invCAA, invCBB, Pl, Bll=None, expend=False):
  '''
  Compute El = invCAA.Pl.invCBB
  '''
  if Bll == None: # Tegmark B-matrix useless so far)
    Bll = np.diagflat((np.ones(len(Pl))))

  lmax = len(Pl)*2**int(expend)
  lrange = np.arange(lmax)
  npix = len(invCAA)
  if expend: # triangular shape ds_dcb
    for l in lrange:
      Pl[l] = np.dot(np.dot(invCAA, Expd(Pl,l)), invCBB).reshape((npix,npix ))
  else:
    for l in lrange:
      Pl[l] = np.dot(np.dot(invCAA, Pl[l]), invCBB).reshape((npix,npix ))

def CrossFisherMatrix(El, Pl, expend=False):
  '''
  Compute cross (or auto) fisher matrix Fll = Trace[invCAA.Pl.invCBB.Pl] = Trace[El.Pl]
  '''
  lmax = len(Pl)*2**int(expend)
  lrange = np.arange(lmax)
  if expend:
    FAB = np.array([np.sum(El[il]*Expd(Pl,jl)) for il in lrange for jl in lrange]).reshape(lmax,lmax) #pas de transpose car symm
  else:
    FAB = np.array([np.sum(El[il]*Pl[jl]) for il in lrange for jl in lrange]).reshape(lmax,lmax) #pas de transpose car symm
  return FAB

def CrossWindowFunction(El, Pl):
  '''
  Compute Tegmark cross (or auto) window matrix Wll = Trace[invCAA.Pl.invCBB.Pl] = Trace[El.Pl]
  Equivalent to Fisher matrix Fll when Tegmark B-matrix = 1
  '''
  lmax = len(El)
  lrange = np.arange((lmax))
  Wll = np.array([np.sum(El[il]*Pl[jl]) for il in lrange for jl in lrange]).reshape(lmax,lmax) #pas de transpose car symm
  return Wll

def CrossWindowFunctionLong(invCAA, invCBB, Pl):
  '''
  Compute Tegmark cross (or auto) window matrix Wll = Trace[invCAA.Pl.invCBB.Pl] = Trace[El.Pl]
  Equivalent to Fisher matrix Fll when Tegmark B-matrix = 1
  '''
  lmax = len(Pl)
  lrange = np.arange((lmax))
  Wll = np.array([np.sum(np.dot(np.dot(invCAA, Pl[il]), invCBB)*Pl[jl]) for il in lrange for jl in lrange]).reshape(lmax,lmax) #pas de transpose car symm
  return Wll

def CrossMisherMatrix(El, CAA, CBB):
  '''
  Compute matrix MAB = Trace[El.CAA.El.CBB] (see paper for definition)
  '''
  lmax = len(El)
  lrange = np.arange(lmax)
  El_CAA = np.array([np.dot(CAA,El[il]) for il in lrange])
  El_CBB = np.array([np.dot(CBB,El[il]) for il in lrange])
  MAB = np.array([np.sum(El_CAA[il]*El_CBB[jl].T) for il in lrange for jl in lrange]).reshape(lmax,lmax)
  return MAB

def CrossGisherMatrix(El, CAB):
  '''
  Compute matrix GAB = Trace[El.CAB.El.CAB] (see paper for definition)
  '''
  lmax = len(El)
  lrange = np.arange(lmax)
  El_CAB = np.array([np.dot(CAB,El[il]) for il in lrange])
  GAB = np.array([np.sum(El_CAB[il]*El_CAB[jl].T) for il in lrange for jl in lrange]).reshape(lmax,lmax)
  return GAB

def CrossGisherMatrixLong(El, CAB):
  '''
  Compute matrix GAB = Trace[El.CAB.El.CAB] (see paper for definition)
  '''
  lmax = len(El)
  lrange = np.arange(lmax)
  GAB = np.array([np.sum(np.dot(CAB,El[il])*np.dot(CAB,El[jl]).T) for il in lrange for jl in lrange]).reshape(lmax,lmax)
  return GAB

def yQuadEstimator(dA, dB, El):
  '''
  Compute pre-estimator 'y' such that Cl = Fll^-1 . yl
  '''
  npix=len(dA)
  lrange = np.arange((len(El)))
  y = np.array([dA.dot(El[l]).dot(dB) for l in lrange])
  return y

def ClQuadEstimator(invW, y):
  '''
  Compute estimator 'Cl' such that Cl = Fll^-1 . yl
  '''
  Cl = np.dot(invW,y)
  return Cl

def biasQuadEstimator(NoiseN, El):
  '''
  Compute bias term bl such that Cl = Fll^-1 . ( yl + bias)
  '''
  lrange = np.arange((len(El)))
  return np.array([np.sum(NoiseN*El[l]) for l in lrange])

def blEstimatorFlat(NoiseN, El):
  '''
  Compute bias term bl such that Cl = Fll^-1 . ( yl + bl)
  Not to be confonded with beam function bl(fwhm)
  '''
  lrange = np.arange((len(El)))

  return np.array([np.sum(NoiseN*np.diag(El[l])) for l in lrange])


def CovAB(invWll, GAB):
  '''
  Compute analytical covariance matrix Cov(Cl, Cl_prime)
  '''
  covAB = np.dot(np.dot(invWll, GAB), invWll.T) + invWll
  return covAB




################################# Simulations tools #################################
def muKarcmin2var(muKarcmin, nside):
  '''
  Return pixel variance for a given nside and noise level [1e-6 K . arcmin]
  '''
  pixarea = hp.nside2pixarea(nside, degrees = True)
  varperpix = (muKarcmin*1e-6/60.)**2/pixarea
  return varperpix

def pixvar2nl(pixvar, nside):
  '''
  Return noise spectrum level for a given nside and  pixel variance
  '''
  return pixvar*4.*np.pi/(12*nside**2.)

def getNl(pixvar, nside, nbins):
  '''
  Return noise spectrum for a given nside and pixel variance
  '''
  return pixvar*4.*np.pi/(12*nside**2.)*np.ones((nbins))


def getstokes(polar=True,temp=False,EBTB=False):
  allStoke=['I','Q','U']
  if EBTB:
    der=['TT','EE','BB','TE','EB','TB']
    ind=[0,1,2,3,4,5]
  else:
    der=['TT','EE','BB','TE']
    ind=[0,1,2,3]
  if  not temp:
    allStoke=['Q','U']
    if EBTB:
      der=['EE','BB','EB']
      ind=[1,2,4]
    else:
      der=['EE','BB']
      ind=[1,2]
  if not polar:
    allStoke=['I']
    der=['TT']
    ind=[0]
  return allStoke, der, ind

def GetBinningMatrix(ellbins, lmax, norm=False,polar=True,temp=False,EBTB=False, verbose=False):
  '''
  Return P and Q matrices such taht Cb = P.Cl and Vbb = P.Vll.Q
  Return ell (total non-binned multipole range)
  Return ellval (binned multipole range)
  '''
  #### define Stokes
  allStoke, der, ind = getstokes(polar,temp,EBTB)
  nder = len(der)

  nbins=len(ellbins)-1
  ellmin=np.array(ellbins[0:nbins])
  ellmax=np.array(ellbins[1:nbins+1])-1
  ell=np.arange(np.min(ellbins),lmax+2)
  maskl=(ell[:-1]<(lmax+2)) & (ell[:-1]>1)

  minell=np.array(ellbins[0:nbins]) # define min
  maxell=np.array(ellbins[1:nbins+1])-1 # and max of a bin
  ellval=(minell+maxell)*0.5

  masklm=[]
  for i in np.arange(nbins):
    masklm.append(((ell[:-1]>=minell[i]) & (ell[:-1]<=maxell[i])))

  allmasklm = nder*[list(masklm)]
  masklM = np.array(sparse.block_diag(allmasklm[:]).toarray())
  binsnorm = np.array(nder*[list(np.arange(minell[0],np.max(ellbins)))]).flatten()

  binsnorm = binsnorm*(binsnorm+1)/2./np.pi
  P = np.array(masklM)*1.
  Q = P.T
  P=P/np.sum(P,1)[:,None]
  if norm:
    P*=binsnorm

  return P, Q, ell, ellval

def GetCorr(F):
  nbins = len(F)
  Corr = np.array([F[i,j]/(F[i,i]*F[j,j])**.5 for i in np.arange(nbins) for j in  np.arange(nbins)]).reshape(nbins, nbins)
  return Corr

def IsInvertible(F):
  print("Cond Numb = ", np.linalg.cond(F), "Matrix eps=", np.finfo(F.dtype).eps)
  return np.linalg.cond(F) > np.finfo(F.dtype).eps





################################# Other tools #################################
def GetSizeNumpies():
  '''
  To be copy/past in ipython to get current memory size taken by numpy arrays
  '''
  import operator
  np_arrays = {k:v for k,v in locals().items() if isinstance(v, np.ndarray)}
  np_arrayssize =  {key : round((np_arrays[key]).nbytes/1024./1024./1024., 4) for key in np_arrays }
  sorted_x = sorted(np_arrayssize.items(), key=operator.itemgetter(1))
  for s in np.arange(len(sorted_x)):
    print(sorted_x[s])
  sumx =  sum(np.array(sorted_x)[:,1].astype(np.float))
  print("total = " + str(sumx)+"Gb")

def ComputeSizeDs_dcb(nside, fsky, deltal=1):
  sizeds_dcb = (2*12*nside**2*fsky)**2*8*2*(3.*nside/deltal)/1024./1024./1024.
  print("size (Gb) = "+str(sizeds_dcb) )
  print("possible reduced size (Gb) = "+str(sizeds_dcb/4) )


def get_colors(num_colors):
    import colorsys
    colors=[]
    ndeg = 250.
    for i in np.arange(0., ndeg, ndeg / num_colors):
        hue = i/360.
        lightness  = 0.5#(50 + np.random.rand() * 10)/100.
        saturation = 0.7#(90 + np.random.rand() * 10)/100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))
    return np.array(colors)

def progress_bar(i,n, dt):
  if n != 1:
    ntot=50
    ndone=ntot*i/(n-1)
    a='\r|'
    for k in np.arange(ndone):
      a += '#'
    for k in np.arange(ntot-ndone):
      a += ' '
    fra = i/(n-1.)
    remain = round(dt/fra*(1-fra))
    a += '| '+str(int(100.*fra))+'%'+" : "+str(remain)+" sec = " +str(round(remain/60.,1)) + " min"
    sys.stdout.write(a)
    sys.stdout.flush()
    if i == n-1:
      sys.stdout.write(' Done. Total time = '+str(np.ceil(dt))+" sec = " +str(round(remain/60.,1)) + " min \n")
      sys.stdout.flush()


def check_symmetric(a, tol=1e-8):
    return np.allclose(a, a.T, atol=tol)


def randomword(length):
   return ''.join(rd.choice(string.lowercase) for i in range(length))



################################# Not used #################################

def cov_from_maps(maps0,maps1):
  sh=np.shape(maps0)
  npix=sh[1]
  nbmc=sh[0]
  covmc=np.zeros((npix,npix))
  mm0=np.mean(maps0,axis=0)
  # print(mm0)
  mm1=np.mean(maps1,axis=0)
  # print(mm1)
  themaps0=np.zeros((nbmc,npix))
  themaps1=np.zeros((nbmc,npix))
  start = timeit.default_timer()
  for i in np.arange(npix):
      progress_bar(i,npix,timeit.default_timer()-start)
      themaps0[:,i]=maps0[:,i]-mm0[i]
      themaps1[:,i]=maps1[:,i]-mm1[i]
  print('hy')
  start = timeit.default_timer()
  for i in np.arange(npix):
      progress_bar(i,npix,timeit.default_timer()-start)
      for j in np.arange(npix):
          covmc[i,j]=np.mean(themaps0[:,i]*themaps1[:,j])
          #covmc[j,i]=covmc[i,j]
  return(covmc)
