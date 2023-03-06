#cython: boundscheck=False, wraparound=False, nonecheck=False
import cython
import numpy as np
cimport numpy as np
from cpython cimport array

"# distutils: language = c"
# distutils: sources = libcov.c

np.import_array()


cdef extern from "libcov.h":
    void set_threads(int n)
    void yCy(int nl, int npixA, int npixB, double * dA, double * dB, double *El, double *Cl)
    
    void build_El_single(int npix, double *P_l, double *invCa, double *invCb, double *E_l)
    void build_Gisher(int nl, int npixA, int npixB, double *C, double *El, double *G)
    void build_El(int nl, int npixA, int npixB, double *Pl, double *invCa, double *invCb, double *El)
    void filter_Pl(int nl, int npixA, int npixB, int npix_ext, double *Pl,
                   double * Pl_out, double * MF_A, double * MF_B,);
    void build_Wll(int nl, int npixA, int npixB, double* El, double* Pl, double* Wll)
    void build_dSdC(int nside, int nstokes, int npix, int inl, long *ispec, long *ellbins, long *ipix, double *bl, double* dSdC)
    int ispec2nspec(long *ispec)
    void dlss( double X, int s1, int s2, int lmax, double *d)
    int polrotangle( double *ri, double *rj, double *cos2a, double *sin2a)


def py_set_threads(n):
    set_threads(n)

def yQuadEstimator(np.ndarray[double, ndim=1, mode="c"] dA,
                   np.ndarray[double, ndim=1, mode="c"] dB,
                   np.ndarray[double, ndim=3, mode="c"] El,):
    nl, npixB, npixA = El.shape[:3]
    yl = np.zeros(nl, dtype=np.float64)
    yCy(nl, npixA, npixB,
        <double*> np.PyArray_DATA(dA),
        <double*> np.PyArray_DATA(dB),
        <double*> np.PyArray_DATA(El),
        <double*> np.PyArray_DATA(yl))
    return yl

def dSdC(nside, nstokes,
         np.ndarray[long, ndim=1, mode="c"] ispec not None,
         np.ndarray[long, ndim=1, mode="c"] ellbins not None,
         np.ndarray[long, ndim=1, mode="c"] ipix not None,
         np.ndarray[double, ndim=1, mode="c"] bl not None):
    npix = ipix.shape[0]
    
    npixtot = npix * nstokes
    nbins = ellbins.shape[0] - 1
    nspec = ispec2nspec(<long *>np.PyArray_DATA(ispec))
    dSdC = np.zeros((nspec * nbins, npixtot, npixtot), dtype=np.float64)
    build_dSdC(nside, nstokes, npix, nbins,
         <long*> np.PyArray_DATA(ispec),
         <long*> np.PyArray_DATA(ellbins),
         <long*> np.PyArray_DATA(ipix),
         <double*> np.PyArray_DATA(bl),
         <double*> np.PyArray_DATA(dSdC))
    return dSdC


def ComputeEl(np.ndarray[double, ndim=2, mode="c"] invCa not None,
              np.ndarray[double, ndim=2, mode="c"] invCb not None,
              np.ndarray[double, ndim=3, mode="c"] Pl not None):
    nl, npixA, npixB = Pl.shape[:3]
    El = np.zeros((nl, npixB, npixA), dtype=np.float64)
    build_El(nl, npixA, npixB,
             <double*> np.PyArray_DATA(Pl),
             <double*> np.PyArray_DATA(invCa),
             <double*> np.PyArray_DATA(invCb),
             <double*> np.PyArray_DATA(El))
    return El

def FilterPl(np.ndarray[double, ndim=3, mode="c"] Pl not None,
             np.ndarray[double, ndim=2, mode="c"] MF_A not None,
             np.ndarray[double, ndim=2, mode="c"] MF_B not None,
             ):
    nl, npix_ext = Pl.shape[:2]
    npixA = MF_A.shape[0]
    npixB = MF_B.shape[0]
    Pl_out = np.zeros((nl, npixA, npixB), dtype=np.float64)
    filter_Pl(nl, npixA, npixB, npix_ext,
              <double*> np.PyArray_DATA(Pl),
              <double*> np.PyArray_DATA(Pl_out),
              <double*> np.PyArray_DATA(MF_A),
              <double*> np.PyArray_DATA(MF_B),)
    return Pl_out

def CrossGisher(np.ndarray[double, ndim=2, mode="c"] C not None,
                np.ndarray[double, ndim=3, mode="c"] El not None):
    """
    compute the diagonal component of the Cl covariance matrix.
    """
    nl, npixB, npixA = El.shape[:3]
    Gl = np.zeros((nl, nl), dtype=np.float64)
    build_Gisher(nl, npixA, npixB,
                 <double *>np.PyArray_DATA(C),
                 <double *>np.PyArray_DATA(El),
                 <double *>np.PyArray_DATA(Gl))
    return Gl

def CrossWindow(np.ndarray[double, ndim=3, mode="c"] El not None,
                np.ndarray[double, ndim=3, mode="c"] Pl not None):
    nl, npixA, npixB = Pl.shape[:3]
    Wll = np.zeros((nl, nl), dtype=np.float64)
    build_Wll(nl, npixA, npixB,
              <double*> np.PyArray_DATA(El),
              <double*> np.PyArray_DATA(Pl),
              <double*> np.PyArray_DATA(Wll))
    return Wll


def _dlss(X, s1, s2, lmax, np.ndarray[double, ndim=1, mode="c"] d not None):
     dlss(X, s1, s2, lmax, <double*> np.PyArray_DATA(d))


def Dlss(double X, int s1, int s2, int lmax):
    d = np.ndarray(lmax+1)
    _dlss(X,s1,s2,lmax,d)
    return d


def _polrotangle(np.ndarray[double, ndim=1, mode="c"] ri,
          np.ndarray[double, ndim=1, mode="c"] rj,
          np.ndarray[double, ndim=1, mode="c"] cos2a,
          np.ndarray[double, ndim=1, mode="c"] sin2a  ):
    polrotangle( <double*> np.PyArray_DATA(ri), 
         <double*> np.PyArray_DATA(rj),
         <double*> np.PyArray_DATA(cos2a),
         <double*> np.PyArray_DATA(sin2a))

def polrotang(ri, rj):
    cos2a = np.ndarray(1, order='C')
    sin2a = np.ndarray(1, order='C')
    _polrotangle( ri, rj, cos2a, sin2a)
    return cos2a[0], sin2a[0]
