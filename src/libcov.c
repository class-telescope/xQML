#include "libcov.h"

void build_dSdC( int nside, int nstokes, int npix, int inl, long *ellbins, long *ipix, double *bl, double* dSdC)
{
  
  const int lmax=ellbins[inl-1];
  const int lmax1 = lmax+1;
  const int ns=3, nspecall = 4;
  const int nspec = nstokes2nspec(nstokes);
  const int npixtot = npix*nstokes;

//   fprintf( stdout, "lmax=%d\n", lmax);
//   fprintf( stdout, "npix=%d\n", npix);
//   fprintf( stdout, "nstokes=%d\n", nstokes);
//   fprintf( stdout, "nspec=%d\n", nspec);

#pragma omp parallel default(none) shared(stdout,inl,nside,npix,nstokes,dSdC,ipix,bl,ellbins)
  {
    int s=0;
    double vr[3], vc[3];
    double **dSdCpix=NULL;
    dSdCpix = (double **) malloc( nspecall*lmax1*sizeof(double));
    for( int il=0; il<nspecall*lmax1; il++) {
      dSdCpix[il] = NULL;
      dSdCpix[il] = (double *) calloc( ns*ns, sizeof(double));
      if( dSdCpix[il] == NULL) EXIT_INFO( -1, "Problem allocation dSdCpix (l=%d)...\n", il);
    }
    
#pragma omp for schedule(dynamic)
    /* loop on local pix to build (3x3) blocks */
    for( int cpix=0; cpix<npix; cpix++) {
      pix2vec_ring(nside, ipix[cpix], vc);

      for( int rpix=0; rpix<npix; rpix++) {
	pix2vec_ring(nside, ipix[rpix], vr);

	QML_compute_dSdC( vr, vc, lmax, nstokes, dSdCpix);

	for( int il=0; il<inl; il++) {
	  int l = ellbins[il];
	  s=0;

	  if( nstokes != 2) {
	    dSdC(s*inl+il,0*npix+cpix,0*npix+rpix) = dSdCpix[0*lmax1+l][0*ns+0] * bl[0*lmax1+l]*bl[0*lmax1+l];  //TT on II
	    s++;
	  }
	  
	  //EE-BB
	  if( nstokes > 1) {
	    int s1=s;
	    int s2=s+1;
	    dSdC(s*inl+il,s1*npix+cpix,s1*npix+rpix) = dSdCpix[1*lmax1+l][1*ns+1] * bl[1*lmax1+l]*bl[1*lmax1+l];  //EE on QQ
	    dSdC(s*inl+il,s2*npix+cpix,s1*npix+rpix) = dSdCpix[1*lmax1+l][2*ns+1] * bl[1*lmax1+l]*bl[1*lmax1+l];  //EE on QU
	    dSdC(s*inl+il,s1*npix+cpix,s2*npix+rpix) = dSdCpix[1*lmax1+l][1*ns+2] * bl[1*lmax1+l]*bl[1*lmax1+l];  //EE on UQ
	    dSdC(s*inl+il,s2*npix+cpix,s2*npix+rpix) = dSdCpix[1*lmax1+l][2*ns+2] * bl[1*lmax1+l]*bl[1*lmax1+l];  //EE on UU
	    s++;
	    
	    dSdC(s*inl+il,s1*npix+cpix,s1*npix+rpix) = dSdCpix[2*lmax1+l][1*ns+1] * bl[2*lmax1+l]*bl[2*lmax1+l];  //BB on QQ
	    dSdC(s*inl+il,s2*npix+cpix,s1*npix+rpix) = dSdCpix[2*lmax1+l][2*ns+1] * bl[2*lmax1+l]*bl[2*lmax1+l];  //BB on QU
	    dSdC(s*inl+il,s1*npix+cpix,s2*npix+rpix) = dSdCpix[2*lmax1+l][1*ns+2] * bl[2*lmax1+l]*bl[2*lmax1+l];  //BB on UQ
	    dSdC(s*inl+il,s2*npix+cpix,s2*npix+rpix) = dSdCpix[2*lmax1+l][2*ns+2] * bl[2*lmax1+l]*bl[2*lmax1+l];  //BB on UU
	    s++;
	  }
	  
	  //TE
	  if( nstokes == 3) {
	    dSdC(s*inl+il,1*npix+cpix,0*npix+rpix) = dSdCpix[3*lmax1+l][1*ns+0] * bl[3*lmax1+l]*bl[3*lmax1+l];  //TE on IQ
	    dSdC(s*inl+il,2*npix+cpix,0*npix+rpix) = dSdCpix[3*lmax1+l][2*ns+0] * bl[3*lmax1+l]*bl[3*lmax1+l];  //TE on IU
	    dSdC(s*inl+il,0*npix+cpix,1*npix+rpix) = dSdCpix[3*lmax1+l][0*ns+1] * bl[3*lmax1+l]*bl[3*lmax1+l];  //TE on QI
	    dSdC(s*inl+il,0*npix+cpix,2*npix+rpix) = dSdCpix[3*lmax1+l][0*ns+2] * bl[3*lmax1+l]*bl[3*lmax1+l];  //TE on UI
	  }
	  
	} /* end loop l */
	
      } /* end loop rpix */
    } /* end loop cpix */

    /* free */
    for( int l=0; l<nspecall*lmax1; l++) free( dSdCpix[l]);
    free( dSdCpix);
  } //end omp parallel


//   return( dSdC);
}






/* compute 3x3 matrix correlation matrix for couple (ipix,jpix) */
void QML_compute_dSdC( double *vr, double *vc, int lmax, int nstokes, double **dSdCpix)
{
  double *pl=NULL, *d20=NULL, *d2p2=NULL, *d2m2=NULL;
  double cos2aij, sin2aij, cos2aji, sin2aji;
  double norm, P02, Q22, R22;
  double cos_chi;
  int l, ns=3;

  /* alloc vectors */
  if( nstokes != 2)
    pl   = (double *) calloc( (lmax+1), sizeof(double));
  if( nstokes > 1) {
    d2p2 = (double *) calloc( (lmax+1), sizeof(double));
    d2m2 = (double *) calloc( (lmax+1), sizeof(double));
  }
  if( nstokes > 2)
    d20  = (double *) calloc( (lmax+1), sizeof(double));
  
  /* generate d_ss'(l) */
  cos_chi = QML_scal_prod( vc, vr);
/*   if( cos_chi > 1 || cos_chi < -1) INFO( "%i angle[%d(%d),%d(%d)] = %e\n", rank, rpix_global, rpix_global/nstokes, cpix_global, cpix_global/nstokes, fabs(cos_chi)-1); */
  if( cos_chi >  1) cos_chi =  1.;
  if( cos_chi < -1) cos_chi = -1.;

/* legendre( cos_chi, 0, lmax, pl); */
  if( nstokes != 2) dlss( cos_chi, 0,  0, lmax,   pl);
  if( nstokes >  1) dlss( cos_chi, 2,  2, lmax, d2p2);
  if( nstokes >  1) dlss( cos_chi, 2, -2, lmax, d2m2);
  if( nstokes >  2) dlss( cos_chi, 2,  0, lmax,  d20);

  /* generate rotation angles */
  if( nstokes > 1) polrotangle( vr, vc, &cos2aij, &sin2aij);
  if( nstokes > 1) polrotangle( vc, vr, &cos2aji, &sin2aji);
  
  /* loop on l */
  for( l=2; l<=lmax; l++) {
    norm = (double)(2*l+1) / (4.*M_PI);

    if( nstokes != 2)
      dSdCpix[0*(lmax+1)+l][0*ns + 0] = norm * pl[l] ;  //TT on II

    if( nstokes > 1) {
      Q22 = ( d2p2[l] + d2m2[l] )/2.;
      R22 = ( d2p2[l] - d2m2[l] )/2.;
      
      dSdCpix[1*(lmax+1)+l][1*ns + 1] = norm * ( cos2aij*cos2aji*Q22 + sin2aij*sin2aji*R22);  //EE on QQ
      dSdCpix[1*(lmax+1)+l][2*ns + 1] = norm * (-cos2aij*sin2aji*Q22 + sin2aij*cos2aji*R22);  //EE on QU
      dSdCpix[1*(lmax+1)+l][1*ns + 2] = norm * (-sin2aij*cos2aji*Q22 + cos2aij*sin2aji*R22);  //EE on UQ
      dSdCpix[1*(lmax+1)+l][2*ns + 2] = norm * ( sin2aij*sin2aji*Q22 + cos2aij*cos2aji*R22);  //EE on UU
      
      dSdCpix[2*(lmax+1)+l][1*ns + 1] = norm * ( cos2aij*cos2aji*R22 + sin2aij*sin2aji*Q22);  //BB on QQ
      dSdCpix[2*(lmax+1)+l][2*ns + 1] = norm * (-cos2aij*sin2aji*R22 + sin2aij*cos2aji*Q22);  //BB on QU
      dSdCpix[2*(lmax+1)+l][1*ns + 2] = norm * (-sin2aij*cos2aji*R22 + cos2aij*sin2aji*Q22);  //BB on UQ
      dSdCpix[2*(lmax+1)+l][2*ns + 2] = norm * ( sin2aij*sin2aji*R22 + cos2aij*cos2aji*Q22);  //BB on UU
    }
    
    if( nstokes > 2) {
      P02 = -d20[l];
      dSdCpix[3*(lmax+1)+l][1*ns + 0] =  norm * P02*cos2aji ;  //TE on IQ
      dSdCpix[3*(lmax+1)+l][2*ns + 0] = -norm * P02*sin2aji ;  //TE on IU
      dSdCpix[3*(lmax+1)+l][0*ns + 1] =  norm * P02*cos2aij ;  //TE on QI
      dSdCpix[3*(lmax+1)+l][0*ns + 2] = -norm * P02*sin2aij ;  //TE on UI
    }

  }

  if( nstokes !=2) free( pl);
  if( nstokes > 1) free( d2p2);
  if( nstokes > 1) free( d2m2);
  if( nstokes > 2) free( d20);
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
  QML_cross_prod( ri, rj, rij);
  norm2 = QML_scal_prod(rij, rij);
  /* pixels are either identical or on diametrically opposite sides of the sky */
  if( norm2 <= 1e-15) {
    *cos2a = 1.;
    *sin2a = 0.;  
    return(OK);
  }
  for( i=0; i<3; i++) rij[i] /= sqrt(norm2);

  /* compute z x ri */
  QML_cross_prod( z, ri, ris);
  norm2 = QML_scal_prod( ris, ris);
  if( norm2 <= 1e-15) {
    /* pixel i is at the pole */
    *cos2a = 1.;
    *sin2a = 0.;  
    return(OK);
  }
  for( i=0; i<3; i++) ris[i] /= sqrt(norm2);

  /* compute cosa, sina */
  QML_cross_prod( rij, ris, rijris);

  cosa = QML_scal_prod( rij, ris);
  sina = QML_scal_prod( rijris, ri);

  /* compute cos2a, sin2a */
  *cos2a = 2.*cosa*cosa - 1.;
  *sin2a = 2.*cosa*sina;
  
  return(OK);
}

void QML_cross_prod( double *v, double *w, double *z)
{
  z[0] = v[1]*w[2] - v[2]*w[1];
  z[1] = v[2]*w[0] - v[0]*w[2];
  z[2] = v[0]*w[1] - v[1]*w[0];
}

double QML_scal_prod( double *v, double *w)
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
/* Compute d_l^ss'                                                       */
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










/*************************************************************************/
int nstokes2nspec( int nstokes)
{
  int nspec=0;
  switch( nstokes) {
  case 1:
    nspec = 1;
    break;
  case 2:
    nspec = 2;
    break;
  case 3:
    nspec = 4;
    break;
  }
  
  return( nspec);
}
/*************************************************************************/



/*************************************************************************/
/* Extract from Healpix                                                  */
/*************************************************************************/

void pix2ang_ring(long nside, long ipix, double *theta, double *phi)
  {
  double z;
  pix2ang_ring_z_phi (nside,ipix,&z,phi);
  *theta=acos(z);
  }


void pix2vec_ring(long nside, long ipix, double *vec)
  {
  double z, phi;
  pix2ang_ring_z_phi (nside,ipix,&z,&phi);
  double stheta=sqrt((1.-z)*(1.+z));
  vec[0]=stheta*cos(phi);
  vec[1]=stheta*sin(phi);
  vec[2]=z;
  }

void pix2ang_ring_z_phi(long nside_, long pix, double *z, double *phi)
  {
  long ncap_=nside_*(nside_-1)*2;
  long npix_=12*nside_*nside_;
  double fact2_ = 4./npix_;
  if (pix<ncap_) /* North Polar cap */
    {
    int iring = (1+isqrt(1+2*pix))>>1; /* counted from North pole */
    int iphi  = (pix+1) - 2*iring*(iring-1);

    *z = 1.0 - (iring*iring)*fact2_;
    *phi = (iphi-0.5) * halfpi/iring;
    }
  else if (pix<(npix_-ncap_)) /* Equatorial region */
    {
    double fact1_  = (nside_<<1)*fact2_;
    int ip  = pix - ncap_;
    int iring = ip/(4*nside_) + nside_; /* counted from North pole */
    int iphi  = ip%(4*nside_) + 1;
    /* 1 if iring+nside is odd, 1/2 otherwise */
    double fodd = ((iring+nside_)&1) ? 1 : 0.5;

    int nl2 = 2*nside_;
    *z = (nl2-iring)*fact1_;
    *phi = (iphi-fodd) * M_PI/nl2;
    }
  else /* South Polar cap */
    {
    int ip = npix_ - pix;
    int iring = (1+isqrt(2*ip-1))>>1; /* counted from South pole */
    int iphi  = 4*iring + 1 - (ip - 2*iring*(iring-1));

    *z = -1.0 + (iring*iring)*fact2_;
    *phi = (iphi-0.5) * halfpi/iring;
    }
  }

int isqrt(int v)
  { return (int)(sqrt(v+0.5)); }
/*************************************************************************/