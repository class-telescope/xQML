/* First try M. Tristram
   2013
*/

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <regex.h>
#include "chealpix.h"
#include "fitsio.h"

#define WRITEOUTPUT 0


#define OK 0
#define NOK 1
#define EXIT_INFO(Y,Z,args...) { fprintf( stdout, "[%s:%d] "Z,__func__, __LINE__, ##args); fflush(stdout); exit(Y); }
#define INFO(Y,args...)      { fprintf( stdout, "[%d] "Y, rank, ##args); fflush(stdout); }
#define INFOROOT(Y,args...)      { if( rank == root) {fprintf( stdout, Y, ##args); fflush(stdout);} }

#ifndef DBL_MAX
#define DBL_MAX            1.79769313486231470e+308
#endif
#ifndef MAX
#define MAX(x,y) (x>y ? x : y) /* Maximum of two arguments */
#endif

static char LO='L', UP='U', NO='N', TR='T';
/* static char RowMajor='R', ColMajor='C'; */
int iminus1=-1, izero=0, ione=1;
double dzero=0., done=1.;

void legendre( double X, int m, int lmax, double *P);
void dlss( double X, int s1, int s2, int lmax, double *d);
double fact(int n);
double my_scal_prod( double *v, double *w);
void my_cross_prod( double *v, double *w, double *z);
int polrotangle( double *ri, double *rj, double *cos2a, double *sin2a);
void polrotmatrix( double *ri, double *rj, double *m);
void compute_dSdC( double *vr, double *vc, int lmax, int nstokes, double **dSdCpix);

#define nside2npix(X) (long)12*(long)X*(long)X

int rank;
int root=0;
int myprow, mypcol, rblocksize, cblocksize;
int local_melem, local_velem;
int mdesc[9], vdesc[9];
char *tagname[6] = {"TT","EE","BB","TE","TB","EB"};






/* compute 3x3 matrix correlation matrix for couple (ipix,jpix) */
void compute_dSdC( double *vr, double *vc, int lmax, int nstokes, double **dSdCpix)
{
  double *pl=NULL, *d20=NULL, *d2p2=NULL, *d2m2=NULL;
  double cos2aij, sin2aij, cos2aji, sin2aji;
  double norm, P02, Q22, R22;
  double cos_chi;
  int l, nspec;

  nspec = (nstokes > 1) ? 4 : 1;

  /* alloc vectors */
  pl = (double *) calloc( nspec*(lmax+1), sizeof(double));
  if( nstokes > 1) {
    d20  = (double *) calloc( nspec*(lmax+1), sizeof(double));
    d2p2 = (double *) calloc( nspec*(lmax+1), sizeof(double));
    d2m2 = (double *) calloc( nspec*(lmax+1), sizeof(double));
  }

  /* generate d_ss'(l) */
  cos_chi = my_scal_prod( vc, vr);
/*   if( cos_chi > 1 || cos_chi < -1) INFO( "%i angle[%d(%d),%d(%d)] = %e\n", rank, rpix_global, rpix_global/nstokes, cpix_global, cpix_global/nstokes, fabs(cos_chi)-1); */
  if( cos_chi >  1) cos_chi =  1.;
  if( cos_chi < -1) cos_chi = -1.;

/* legendre( cos_chi, 0, lmax, pl); */
  dlss( cos_chi, 0,  0, lmax, pl);
  if( nstokes > 1) dlss( cos_chi, 2,  0, lmax, d20 );
  if( nstokes > 1) dlss( cos_chi, 2,  2, lmax, d2p2);
  if( nstokes > 1) dlss( cos_chi, 2, -2, lmax, d2m2);

  /* generate rotation angles */
  if( nstokes > 1) polrotangle( vr, vc, &cos2aij, &sin2aij);
  if( nstokes > 1) polrotangle( vc, vr, &cos2aji, &sin2aji);

  /* loop on l */
  for( l=2; l<=lmax; l++) {
    norm = (double)(2*l+1) / (4.*M_PI);

    dSdCpix[0*(lmax+1)+l][0*nstokes + 0] = norm * pl[l] ;  //TT on II

    if( nstokes > 1) {
      P02 = -d20[l];
      Q22 = ( d2p2[l] + d2m2[l] )/2.;
      R22 = ( d2p2[l] - d2m2[l] )/2.;

      dSdCpix[1*(lmax+1)+l][1*nstokes + 1] = norm * ( cos2aij*cos2aji*Q22 + sin2aij*sin2aji*R22);  //EE on QQ
      dSdCpix[1*(lmax+1)+l][2*nstokes + 1] = norm * (-cos2aij*sin2aji*Q22 + sin2aij*cos2aji*R22);  //EE on QU
      dSdCpix[1*(lmax+1)+l][1*nstokes + 2] = norm * (-sin2aij*cos2aji*Q22 + cos2aij*sin2aji*R22);  //EE on UQ
      dSdCpix[1*(lmax+1)+l][2*nstokes + 2] = norm * ( sin2aij*sin2aji*Q22 + cos2aij*cos2aji*R22);  //EE on UU

      dSdCpix[2*(lmax+1)+l][1*nstokes + 1] = norm * ( cos2aij*cos2aji*R22 + sin2aij*sin2aji*Q22);  //BB on QQ
      dSdCpix[2*(lmax+1)+l][2*nstokes + 1] = norm * (-cos2aij*sin2aji*R22 + sin2aij*cos2aji*Q22);  //BB on QU
      dSdCpix[2*(lmax+1)+l][1*nstokes + 2] = norm * (-sin2aij*cos2aji*R22 + cos2aij*sin2aji*Q22);  //BB on UQ
      dSdCpix[2*(lmax+1)+l][2*nstokes + 2] = norm * ( sin2aij*sin2aji*R22 + cos2aij*cos2aji*Q22);  //BB on UU

      dSdCpix[3*(lmax+1)+l][1*nstokes + 0] =  norm * P02*cos2aji ;  //TE on IQ
      dSdCpix[3*(lmax+1)+l][2*nstokes + 0] = -norm * P02*sin2aji ;  //TE on IU
      dSdCpix[3*(lmax+1)+l][0*nstokes + 1] =  norm * P02*cos2aij ;  //TE on QI
      dSdCpix[3*(lmax+1)+l][0*nstokes + 2] = -norm * P02*sin2aij ;  //TE on UI
    }

  }

  free( pl);
  if( nstokes > 1) free( d20);
  if( nstokes > 1) free( d2p2);
  if( nstokes > 1) free( d2m2);
}




/*************************************************************************/
/* Polar Rotation Angle                                                  */
/*************************************************************************/
int polrotangle( double *ri, double *rj, double *cos2a, double *sin2a)
{
  int i;
  double norm2, cosa, sina;
  double rij[3], ris[3], rijris[3];
  double z[3] = {0.,0.,1.};

  /* compute ri x rj */
  my_cross_prod( ri, rj, rij);
  norm2 = my_scal_prod(rij, rij);
  /* pixels are either identical or on diametrically opposite sides of the sky */
  if( norm2 <= 1e-15) {
    *cos2a = 1.;
    *sin2a = 0.;  
    return(OK);
  }
  for( i=0; i<3; i++) rij[i] /= sqrt(norm2);

  /* compute z x ri */
  my_cross_prod( z, ri, ris);
  norm2 = my_scal_prod( ris, ris);
  if( norm2 <= 1e-15) {
    /* pixel i is at the pole */
    *cos2a = 1.;
    *sin2a = 0.;  
    return(OK);
  }
  for( i=0; i<3; i++) ris[i] /= sqrt(norm2);


  /* compute cosa, sina */
  my_cross_prod( rij, ris, rijris);

  cosa = my_scal_prod( rij, ris);
  sina = my_scal_prod( rijris, ri);

  /* compute cos2a, sin2a */
  *cos2a = 2.*cosa*cosa - 1.;
  *sin2a = 2.*cosa*sina;
  
  return(OK);
}

void my_cross_prod( double *v, double *w, double *z)
{
  z[0] = v[1]*w[2] - v[2]*w[1];
  z[1] = v[2]*w[0] - v[0]*w[2];
  z[2] = v[0]*w[1] - v[1]*w[0];
}

double my_scal_prod( double *v, double *w)
{
  double s;

  s = v[0]*w[0] + v[1]*w[1] + v[2]*w[2];

  return( s);
  
}
/*************************************************************************/




/*************************************************************************/
/* Compute Legendre                                                      */
/*************************************************************************/
void legendre( double X, int m, int lmax, double *P)
{
  int l, i;
  
  if( (m > lmax) || (m < 0) )
    EXIT_INFO( -1, "Wrong m value : %d\n", m);
  
  /* compute Pmm(X) */
  P[m] = 1.0;
  for( i=1; i<=m; i++) {
    P[m] = -(2.*i+1) * sqrt(1.-X*X) * P[m];
  }

  /* compute Pmm1(X) */
  P[m+1] = X*(double)(2*m+1)*P[m];
  
  /* Compute Plm(X) */
  for( l=m+2; l<=lmax; l++)
    P[l] = ( (double)(2*l-1)*X*P[l-1] - (double)(l+m-1)*P[l-2] ) / (double)(l-m);

}
/*************************************************************************/





/*************************************************************************/
/* Compute d_l^ss'                                                        */
/*************************************************************************/
void dlss( double X, int s1, int s2, int lmax, double *d)
{
  int l, l1, sign;
  double rhoSSL, rhoSSL1;
  
  if( s1 < abs(s2) )
    EXIT_INFO( -1, "Wrong s values (requested s<abs(s')) : s=%d s'=%d\n", s1, s2);

  /* init l=s1 */
  sign = ( (s1+s2) & 1 ) ? -1 : 1;
  d[s1] = sign / pow(2.,s1) * sqrt( fact(2*s1) / fact(s1+s2) / fact(s1-s2) ) * pow( (1.+X), (s1+s2)/2.) * pow( (1.-X), (s1-s2)/2.);  

  /* init l=s1+1 */
  l1 = s1+1;
  rhoSSL1 = sqrt( ((double)l1*l1 - (double)s1*s1) * ((double)l1*l1 - (double)s2*s2) ) / (double)l1;
  d[s1+1] = (double)(2*s1+1)*(X-(double)s2/(double)(s1+1)) * d[s1] / rhoSSL1;

  /* Compute dss'_l(X) */
  for( l=s1+1; l<lmax; l++) {
    l1 = l+1;
    rhoSSL  = sqrt( ((double)l*l   - (double)s1*s1) * ((double)l*l   - (double)s2*s2) ) / (double)l;
    rhoSSL1 = sqrt( ((double)l1*l1 - (double)s1*s1) * ((double)l1*l1 - (double)s2*s2) ) / (double)l1;
    d[l+1] = ( (double)(2*l+1)*(X-(double)s1*s2/(double)l/(double)l1)*d[l] - rhoSSL*d[l-1] ) / rhoSSL1;    
  }

}

double fact( int n)
{
     double f = 1.;
     while( n > 1) f *= (double)(n--);
     return f;
}
/*************************************************************************/




