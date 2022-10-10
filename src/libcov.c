#include "libcov.h"

void build_Gisher(int nl, int npix, double *C, double *El, double *G){
    int64_t npixtot = npix*npix;
    double *El_CAB =(double *)malloc(sizeof(double)*nl*npixtot);
    openblas_set_num_threads(1);
    #pragma omp parallel
    {
        #pragma omp for
        for(int l=0; l<nl; l++){
            cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper,
                npix, npix,
                1, C, npix,
                &El[l*npixtot], npix,
                0, &El_CAB[l*npixtot], npix);
        }

        #pragma omp for
        for(int l1=0; l1<nl; l1++){
            for(int l2=l1; l2<nl; l2++){
                for(int i=0; i<npix; i++){
                for(int j=0; j<npix; j++){
                    G[l1*nl+l2] += El_CAB[l1*npixtot+i*npix+j]*El_CAB[l2*npixtot+j*npix+i];
                }
                }
                if(l2>l1){
                    G[l2*nl+l1] = G[l1*nl+l2];
                }
            }
        }
    }
    free(El_CAB);
}

void build_El(int nl, int npix, double *Pl, double *invCa, double *invCb, double *El){
    int64_t npixtot = npix*npix;
    openblas_set_num_threads(1);
    #pragma omp parallel
    {
        double *tmp = (double *)malloc(sizeof(double)*npixtot);
        #pragma omp for
        for(int i=0; i<nl; i++){
            cblas_dsymm(CblasRowMajor, CblasLeft, CblasUpper,
                        npix, npix,
                        1, invCa, npix,
                        &Pl[i*npixtot], npix,
                        0, tmp, npix);
            cblas_dsymm(CblasRowMajor, CblasRight, CblasUpper,
                        npix, npix,
                        1, invCb, npix,
                        tmp, npix,
                        0, &El[i*npixtot], npix);
        }
        free(tmp);
   }
}

//problem of precision wrt python...
void build_Wll(int nl, int npix, double* El, double* Pl, double* Wll)
{
  int64_t npixtot = npix*npix;
  
  memset(Wll, 0., (nl*nl) * sizeof(double));

  #pragma omp parallel default(none) shared(nl, npixtot, El, Pl, Wll)
  {
    #pragma omp for schedule(dynamic)
    for( int l1=0; l1<nl; l1++) {
      for( int l2=0; l2<nl; l2++) {

        for( int p=0; p<npixtot; p++)
            Wll[l1*nl+l2] += El[l1*npixtot+p]*Pl[l2*npixtot+ p];
      } //loop l2
    } //loop l1
    
  } //end omp parallel
}


void build_dSdC(int nside, int nstokes, int npix, int nbin, long *ispec, long *ellbins, long *ipix, double *bl, double* dSdC)
{
  const int nspec = ispec2nspec(ispec);
  const int64_t npixtot = npix*nstokes;

  int64_t ntot = nspec*nbin*npixtot*npixtot;
  memset( dSdC, 0., ntot * sizeof(double));

  #pragma omp parallel default(none) shared(stdout,nbin,nside,npix,nstokes,dSdC,ipix,bl,ellbins,ispec, npixtot)
  {
    const int lmax=ellbins[nbin]-1;
    const int lmax1 = lmax+1;
    const int ns=3, nspecall = 6;
    int s=0;
    double vr[3], vc[3];
    double **dSdCpix=NULL;
    dSdCpix = (double **) malloc( nspecall*lmax1*sizeof(double));
    for(int il=0; il<nspecall*lmax1; il++) {
      dSdCpix[il] = NULL;
      dSdCpix[il] = (double *) calloc( ns*ns, sizeof(double));
      if(dSdCpix[il] == NULL) EXIT_INFO( -1, "Problem allocation dSdCpix (l=%d)...\n", il);
    }
    
    int sI = 0, sQ = 1, sU = 2;
    if(nstokes == 2) { sQ = 0; sU = 1; }
    
    #pragma omp for schedule(dynamic)
    /* loop on local pix to build (3x3) blocks */
    for(int cpix=0; cpix<npix; cpix++) {
      pix2vec_ring(nside, ipix[cpix], vc);

      for(int rpix=0; rpix<npix; rpix++) {
        pix2vec_ring(nside, ipix[rpix], vr);
        QML_compute_dSdC( vr, vc, lmax, ispec, dSdCpix);
        for(int ib=0; ib<nbin; ib++) {
        for(int l=ellbins[ib]; l<=ellbins[ib+1]-1; l++) {
            s=0;

        if( ispec[0] == 1) {
          dSdC(s*nbin+ib,sI*npix+cpix,sI*npix+rpix) += dSdCpix[0*lmax1+l][0*ns+0] * bl[0*lmax1+l]*bl[0*lmax1+l];  //TT on II
          s++;
        }

        //EE-BB
        if( ispec[1] == 1 || ispec[2] == 1) {
          dSdC(s*nbin+ib,sQ*npix+cpix,sQ*npix+rpix) += dSdCpix[1*lmax1+l][1*ns+1] * bl[1*lmax1+l]*bl[1*lmax1+l];  //EE on QQ
          dSdC(s*nbin+ib,sU*npix+cpix,sQ*npix+rpix) += dSdCpix[1*lmax1+l][2*ns+1] * bl[1*lmax1+l]*bl[1*lmax1+l];  //EE on QU
          dSdC(s*nbin+ib,sQ*npix+cpix,sU*npix+rpix) += dSdCpix[1*lmax1+l][1*ns+2] * bl[1*lmax1+l]*bl[1*lmax1+l];  //EE on UQ
          dSdC(s*nbin+ib,sU*npix+cpix,sU*npix+rpix) += dSdCpix[1*lmax1+l][2*ns+2] * bl[1*lmax1+l]*bl[1*lmax1+l];  //EE on UU
          s++;

          dSdC(s*nbin+ib,sQ*npix+cpix,sQ*npix+rpix) += dSdCpix[2*lmax1+l][1*ns+1] * bl[2*lmax1+l]*bl[2*lmax1+l];  //BB on QQ
          dSdC(s*nbin+ib,sU*npix+cpix,sQ*npix+rpix) += dSdCpix[2*lmax1+l][2*ns+1] * bl[2*lmax1+l]*bl[2*lmax1+l];  //BB on QU
          dSdC(s*nbin+ib,sQ*npix+cpix,sU*npix+rpix) += dSdCpix[2*lmax1+l][1*ns+2] * bl[2*lmax1+l]*bl[2*lmax1+l];  //BB on UQ
          dSdC(s*nbin+ib,sU*npix+cpix,sU*npix+rpix) += dSdCpix[2*lmax1+l][2*ns+2] * bl[2*lmax1+l]*bl[2*lmax1+l];  //BB on UU
          s++;
        }

        //TE
        if( ispec[3] == 1) {
          dSdC(s*nbin+ib,sI*npix+cpix,sQ*npix+rpix) += dSdCpix[3*lmax1+l][0*ns+1] * bl[0*lmax1+l]*bl[1*lmax1+l];  //TE on IQ
          dSdC(s*nbin+ib,sI*npix+cpix,sU*npix+rpix) += dSdCpix[3*lmax1+l][0*ns+2] * bl[0*lmax1+l]*bl[1*lmax1+l];  //TE on IU
          dSdC(s*nbin+ib,sQ*npix+cpix,sI*npix+rpix) += dSdCpix[3*lmax1+l][1*ns+0] * bl[0*lmax1+l]*bl[1*lmax1+l];  //TE on QI
          dSdC(s*nbin+ib,sU*npix+cpix,sI*npix+rpix) += dSdCpix[3*lmax1+l][2*ns+0] * bl[0*lmax1+l]*bl[1*lmax1+l];  //TE on UI
          s++;
        }

        //TB
        if( ispec[4] == 1) {
          dSdC(s*nbin+ib,sI*npix+cpix,sQ*npix+rpix) += dSdCpix[4*lmax1+l][0*ns+1] * bl[0*lmax1+l]*bl[2*lmax1+l];  //TB on IQ
          dSdC(s*nbin+ib,sI*npix+cpix,sU*npix+rpix) += dSdCpix[4*lmax1+l][0*ns+2] * bl[0*lmax1+l]*bl[2*lmax1+l];  //TB on IU
          dSdC(s*nbin+ib,sQ*npix+cpix,sI*npix+rpix) += dSdCpix[4*lmax1+l][1*ns+0] * bl[0*lmax1+l]*bl[2*lmax1+l];  //TB on QI
          dSdC(s*nbin+ib,sU*npix+cpix,sI*npix+rpix) += dSdCpix[4*lmax1+l][2*ns+0] * bl[0*lmax1+l]*bl[2*lmax1+l];  //TB on UI
          s++;
        }

        //EB
        if( ispec[5] == 1) {
          dSdC(s*nbin+ib,sQ*npix+cpix,sQ*npix+rpix) += dSdCpix[5*lmax1+l][1*ns+1] * bl[1*lmax1+l]*bl[2*lmax1+l];  //EB on QQ
          dSdC(s*nbin+ib,sQ*npix+cpix,sU*npix+rpix) += dSdCpix[5*lmax1+l][1*ns+2] * bl[1*lmax1+l]*bl[2*lmax1+l];  //EB on QU
          dSdC(s*nbin+ib,sU*npix+cpix,sQ*npix+rpix) += dSdCpix[5*lmax1+l][2*ns+1] * bl[1*lmax1+l]*bl[2*lmax1+l];  //EB on UQ
          dSdC(s*nbin+ib,sU*npix+cpix,sU*npix+rpix) += dSdCpix[5*lmax1+l][2*ns+2] * bl[1*lmax1+l]*bl[2*lmax1+l];  //EB on UU
          s++;
        }

      } /* end loop l */

    } /* end loop bins */

      } /* end loop rpix */
    } /* end loop cpix */

    /* free */
    for(int l=0; l<nspecall*lmax1; l++) free(dSdCpix[l]);
    free(dSdCpix);

  } //end omp parallel

}


/* compute 3x3 matrix correlation matrix for couple (ipix,jpix) */
void QML_compute_dSdC(double *vr, double *vc, int lmax, long *ispec, double **dSdCpix)
{
  double *pl=NULL, *d20=NULL, *d2p2=NULL, *d2m2=NULL;
  double cos2aij, sin2aij, cos2aji, sin2aji;
  double norm, P02, Q22, R22;
  double cos_chi;
  int l, ns=3;
  int spin0=0, spin2=0, spin02=0;

  /* alloc vectors */
  if( ispec[0] == 1) spin0 = 1;
  if( ispec[1] == 1 || ispec[2] == 1 || ispec[5] == 1) spin2 = 1;
  if( ispec[3] == 1 || ispec[4] == 1) spin02 = 1;
  
  if( spin0 ) pl   = (double *) calloc( (lmax+1), sizeof(double));
  if( spin2 ) d2p2 = (double *) calloc( (lmax+1), sizeof(double));
  if( spin2 ) d2m2 = (double *) calloc( (lmax+1), sizeof(double));
  if( spin02) d20  = (double *) calloc( (lmax+1), sizeof(double));
  
  /* generate d_ss'(l) */
  cos_chi = QML_scal_prod( vc, vr);
/*   if( cos_chi > 1 || cos_chi < -1) INFO( "%i angle[%d(%d),%d(%d)] = %e\n", rank, rpix_global, rpix_global/nstokes, cpix_global, cpix_global/nstokes, fabs(cos_chi)-1); */
  if( cos_chi >  1) cos_chi =  1.;
  if( cos_chi < -1) cos_chi = -1.;

/* legendre( cos_chi, 0, lmax, pl); */
  if( spin0 ) dlss( cos_chi, 0,  0, lmax,   pl);
  if( spin2 ) dlss( cos_chi, 2,  2, lmax, d2p2);
  if( spin2 ) dlss( cos_chi, 2, -2, lmax, d2m2);
  if( spin02) dlss( cos_chi, 2,  0, lmax,  d20);

  /* generate rotation angles */
  if( spin2 || spin02) polrotangle( vr, vc, &cos2aij, &sin2aij);
  if( spin2 || spin02) polrotangle( vc, vr, &cos2aji, &sin2aji);
  
  /* loop on l */
  for( l=2; l<=lmax; l++) {
    norm = (double)(2*l+1) / (4.*M_PI);

    if( ispec[0] == 1)
      dSdCpix[0*(lmax+1)+l][0*ns + 0] = norm * pl[l] ;  //TT on II

    if( ispec[1] == 1 || ispec[2] == 1) {
      Q22 = norm * ( d2p2[l] + d2m2[l] )/2.;
      R22 = norm * ( d2p2[l] - d2m2[l] )/2.;
      
      dSdCpix[1*(lmax+1)+l][1*ns + 1] = ( cos2aij*cos2aji*Q22 + sin2aij*sin2aji*R22);  //EE on QQ
      dSdCpix[1*(lmax+1)+l][2*ns + 1] = (-cos2aij*sin2aji*Q22 + sin2aij*cos2aji*R22);  //EE on QU
      dSdCpix[1*(lmax+1)+l][1*ns + 2] = (-sin2aij*cos2aji*Q22 + cos2aij*sin2aji*R22);  //EE on UQ
      dSdCpix[1*(lmax+1)+l][2*ns + 2] = ( sin2aij*sin2aji*Q22 + cos2aij*cos2aji*R22);  //EE on UU
      
      dSdCpix[2*(lmax+1)+l][1*ns + 1] = ( cos2aij*cos2aji*R22 + sin2aij*sin2aji*Q22);  //BB on QQ
      dSdCpix[2*(lmax+1)+l][2*ns + 1] = (-cos2aij*sin2aji*R22 + sin2aij*cos2aji*Q22);  //BB on QU
      dSdCpix[2*(lmax+1)+l][1*ns + 2] = (-sin2aij*cos2aji*R22 + cos2aij*sin2aji*Q22);  //BB on UQ
      dSdCpix[2*(lmax+1)+l][2*ns + 2] = ( sin2aij*sin2aji*R22 + cos2aij*cos2aji*Q22);  //BB on UU
    }
    
    if( ispec[3] == 1) {
      P02 = -norm*d20[l];
      dSdCpix[3*(lmax+1)+l][1*ns + 0] =   P02*cos2aji ;  //TE on IQ
      dSdCpix[3*(lmax+1)+l][2*ns + 0] = - P02*sin2aji ;  //TE on IU
      dSdCpix[3*(lmax+1)+l][0*ns + 1] =   P02*cos2aij ;  //TE on QI
      dSdCpix[3*(lmax+1)+l][0*ns + 2] = - P02*sin2aij ;  //TE on UI
    }

    if( ispec[4] == 1) {
      P02 = -norm*d20[l];
      dSdCpix[4*(lmax+1)+l][1*ns + 0] = P02*sin2aji ;  //TB on IQ
      dSdCpix[4*(lmax+1)+l][2*ns + 0] = P02*cos2aji ;  //TB on IU
      dSdCpix[4*(lmax+1)+l][0*ns + 1] = P02*sin2aij ;  //TB on QI
      dSdCpix[4*(lmax+1)+l][0*ns + 2] = P02*cos2aij ;  //TB on UI
    }

    if( ispec[5] == 1) {
      Q22 = norm * ( d2p2[l] + d2m2[l] )/2.;
      R22 = norm * ( d2p2[l] - d2m2[l] )/2.;      
      dSdCpix[5*(lmax+1)+l][1*ns + 1] = ( cos2aij*sin2aji*Q22 - sin2aij*cos2aji*R22 + sin2aij*cos2aji*Q22 - cos2aij*sin2aji*R22);  //EB on QQ
      dSdCpix[5*(lmax+1)+l][1*ns + 2] = ( cos2aij*cos2aji*Q22 + sin2aij*sin2aji*R22 - sin2aij*sin2aji*Q22 - cos2aij*cos2aji*R22);  //EB on QU
      dSdCpix[5*(lmax+1)+l][2*ns + 1] = ( cos2aij*cos2aji*Q22 + sin2aij*sin2aji*R22 - sin2aij*sin2aji*Q22 - cos2aij*cos2aji*R22);  //EB on UQ
      dSdCpix[5*(lmax+1)+l][2*ns + 2] = (-cos2aij*sin2aji*Q22 + sin2aij*cos2aji*R22 - sin2aij*cos2aji*Q22 + cos2aij*sin2aji*R22);  //EB on UU
    }

  }

  if( spin0 ) free( pl  );
  if( spin2 ) free( d2p2);
  if( spin2 ) free( d2m2);
  if( spin02) free( d20 );
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
int ispec2nspec(long *ispec)
{
  int nspec=0;

  //force TT & EE for TE
  if( ispec[3] == 1) ispec[0] = ispec[1] = 1;

  //force TT & BB for TB
  if( ispec[4] == 1) ispec[0] = ispec[2] = 1;

  //force EE & BB
  if( ispec[1] == 1) ispec[1] = ispec[2] = 1;
  if( ispec[2] == 1) ispec[1] = ispec[2] = 1;  

  //force EE & BB for EB
  if( ispec[5] == 1) ispec[1] = ispec[2] = 1;

  for( int i=0; i<6; i++)
    if( ispec[i]) nspec++;

  return(nspec);
}

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
