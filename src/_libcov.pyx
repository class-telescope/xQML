#cython: boundscheck=False, wraparound=False, nonecheck=False
import cython
cimport numpy as np

from cpython cimport array

"# distutils: language = c"
# distutils: sources = libcov.c

np.import_array()


cdef extern from "libcov.h":
     void build_dSdC( int nside, int nstokes, int npix, int inl, long *ellbins, long *ipix, double *bl, double* dSdC)


def dSdC( nside, nstokes, 
	  np.ndarray[long, ndim=1, mode="c"] ellbins not None, 
	  np.ndarray[long, ndim=1, mode="c"] ipix not None, 
	  np.ndarray[double, ndim=1, mode="c"] bl not None, 
	  np.ndarray[double, ndim=1, mode="c"] dSdC not None):
     build_dSdC( nside, nstokes, ipix.shape[0], ellbins.shape[0]-1,
		 <long*> np.PyArray_DATA(ellbins),
		 <long*> np.PyArray_DATA(ipix), 
		 <double*> np.PyArray_DATA(bl),
		 <double*> np.PyArray_DATA(dSdC))

