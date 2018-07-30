"""
Set of routines to ...

Author: Vanneste
"""
from __future__ import division

import sys
import timeit

import numpy as np

def GetSizeNumpies():
    """
    To be copy/past in ipython to get current memory size taken by numpy arrays
    """
    import operator
    np_arrays = {k:v for k,v in locals().items() if isinstance(v, np.ndarray)}
    np_arrayssize =  {key : round((np_arrays[key]).nbytes/1024./1024./1024., 4) for key in np_arrays }
    sorted_x = sorted(np_arrayssize.items(), key=operator.itemgetter(1))
    for s in np.arange(len(sorted_x)):
        print(sorted_x[s])
    sumx =  sum(np.array(sorted_x)[:,1].astype(np.float))
    print("total = " + str(sumx)+"Gb")

def ComputeSizeDs_dcb(nside, fsky, deltal=1):
    """
    ???

    Parameters
    ----------
    nside : ???
        ???

    Returns
    ----------
    ???
    """
    sizeds_dcb = (2*12*nside**2*fsky)**2*8*2*(3.*nside/deltal)/1024./1024./1024.
    print("size (Gb) = "+str(sizeds_dcb) )
    print("possible reduced size (Gb) = "+str(sizeds_dcb/4) )


def get_colors(num_colors):
    """
    ???

    Parameters
    ----------
    num_colors : ???
        ???

    Returns
    ----------
    ???
    """
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
    """
    ???

    Parameters
    ----------
    i : ???
        ???

    Returns
    ----------
    ???
    """
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
    """
    ???

    Parameters
    ----------
    a : ???
        ???

    Returns
    ----------
    ???
    """
    return np.allclose(a, a.T, atol=tol)


def randomword(length):
    """
    ???

    Parameters
    ----------
    length : ???
        ???

    Returns
    ----------
    ???
    """
    return ''.join(rd.choice(string.lowercase) for i in range(length))



################################# Not used #################################

def cov_from_maps(maps0,maps1):
    """
    ???

    Parameters
    ----------
    maps0 : ???
        ???

    Returns
    ----------
    ???
    """
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
