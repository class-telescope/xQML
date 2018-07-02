xQML

a quadratic Power Spectrum estimator based on cross-correlation between maps.
[Vanneste et al.]

This is the generalisation to cross-correlation of QML methods
[Tegmark Physical Review D 55, 5895 (1997), Tegmark and de Oliveira-Costa, Physical Review D 64 (2001)]


<br>
<br>

The code is UNDER DEVELOPMENT !

For now it is a simple python librairie.
We are working to externalise CPU intensive routines and use OpenMP.

Librairies needed are :
- scipy
- numpy
- healpy


List of several routines:
Pl, S = compute_ds_dcb(ellbins,nside,ipok,bl, clth, Slmax, polar=,temp=,EBTB=, pixwining=, timing=, Sonly=)
El = El(invCAA, invCBB, Pl, Bll=None)
W = CrossWindowFunction(El, Pl)

or directly without El (but longer)
W = CrossWindowFunctionLong(invCAA, invCBB, Pl)

G = CrossGisherMatrix(El, CAB)
V = CovAB(invW, GAB)

yl = yQuadEstimator(dA, dB, El)
cl = ClQuadEstimator(invW, yl)
