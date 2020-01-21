#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
/* #include <chealpix.h> */

#define OK 0
#define NOK 1
#define EXIT_INFO(Y,Z,args...) { fprintf( stdout, "[%s:%d] "Z,__func__, __LINE__, ##args); fflush(stdout); exit(Y); }
#define INFO(Y,args...)      { fprintf( stdout, "[%d] "Y, rank, ##args); fflush(stdout); }

#define dSdC(L,I,J) dSdC[ (L)*(npixtot*npixtot) + (I)*npixtot + (J)]
#define halfpi M_PI/2.

void build_dSdC( int nside, int nstokes, int npix, int inl, long *ispec, long *ellbins, long *ipix, double *bl, double* dSdC);
/**
    Compute the Pl = dS/dCl matrices.

    Parameters
    ----------
    ellbins : array of floats
        Lowers bounds of bins. Example : taking ellbins = (2, 3, 5) will
        compute the spectra for bins = (2, 3.5).
    nside : int
        Healpix map resolution
    nstokes : int
        number of stoks parameters( 1=Temp, 2=Polar, 3=T+P)
    npix : int
        number of pixels
    ipix : array of ints
        Healpy pixels numbers considered
    bl : 1D array of floats
        Beam window function

    Return
    ------
    dSdC : 2D array double
        Normalize Legendre polynomials dS/dCl
 */


void QML_compute_dSdC( double *vr, double *vc, int lmax, long *ispec, double **dSdCpix);
int polrotangle( double *ri, double *rj, double *cos2a, double *sin2a);
void QML_cross_prod( double *v, double *w, double *z);
double QML_scal_prod( double *v, double *w);
void legendre( double X, int m, int lmax, double *P);
void dlss( double X, int s1, int s2, int lmax, double *d);
double fact( int n);

int define_lrange( int lmax, int *ils);
int nstokes2nspec( int nstokes);
int ispec2nspec( long *ispec);

void pix2ang_ring(long nside, long ipix, double *theta, double *phi);
void pix2vec_ring(long nside, long ipix, double *vec);
void pix2ang_ring_z_phi(long nside_, long pix, double *z, double *phi);
int isqrt(int v);


