#cython: boundscheck=False, wraparound=False, nonecheck=False
import cython
cimport numpy as np

from cpython cimport array

"# distutils: language = c"
# distutils: sources = libcov.c

np.import_array()


cdef extern from "libcov.h":
     void build_dSdC( int nside, int nstokes, int npix, int inl, long *ispec, long *ellbins, long *ipix, double *bl, double* dSdC)
     void dlss( double X, int s1, int s2, int lmax, double *d)
     int polrotangle( double *ri, double *rj, double *cos2a, double *sin2a)


def dSdC( nside, nstokes, 
	  np.ndarray[long, ndim=1, mode="c"] ispec not None, 
	  np.ndarray[long, ndim=1, mode="c"] ellbins not None, 
	  np.ndarray[long, ndim=1, mode="c"] ipix not None, 
	  np.ndarray[double, ndim=1, mode="c"] bl not None, 
	  np.ndarray[double, ndim=1, mode="c"] dSdC not None):
     build_dSdC( nside, nstokes, ipix.shape[0], ellbins.shape[0]-1,
		 <long*> np.PyArray_DATA(ispec),
		 <long*> np.PyArray_DATA(ellbins),
		 <long*> np.PyArray_DATA(ipix), 
		 <double*> np.PyArray_DATA(bl),
		 <double*> np.PyArray_DATA(dSdC))


def _dlss( X, s1, s2, lmax, np.ndarray[double, ndim=1, mode="c"] d not None):
     dlss( X, s1, s2, lmax, <double*> np.PyArray_DATA(d))


def Dlss( double X, int s1, int s2, int lmax):
    d = np.ndarray( lmax+1)
    _dlss(X,s1,s2,lmax,d)
    return( d)


def _polrotangle( np.ndarray[double, ndim=1, mode="c"] ri,
	 	  np.ndarray[double, ndim=1, mode="c"] rj,
		  np.ndarray[double, ndim=1, mode="c"] cos2a,
		  np.ndarray[double, ndim=1, mode="c"] sin2a  ):
    polrotangle( <double*> np.PyArray_DATA(ri), 
		 <double*> np.PyArray_DATA(rj), 
		 <double*> np.PyArray_DATA(cos2a), 
		 <double*> np.PyArray_DATA(sin2a))

def polrotang( ri, rj):
    cos2a = np.ndarray(1, order='C')
    sin2a = np.ndarray(1, order='C')
    _polrotangle( ri, rj, cos2a, sin2a)
    return (cos2a[0], sin2a[0])
