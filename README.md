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


List of several routines:<br>
Pl, S = compute_ds_dcb(ellbins,nside,ipok,bl, clth, Slmax, polar=,temp=,EBTB=, pixwining=, timing=, Sonly=)<br>
El = El(invCAA, invCBB, Pl, Bll=None)<br>
W = CrossWindowFunction(El, Pl)<br>

or directly without El (but longer)<br>
W = CrossWindowFunctionLong(invCAA, invCBB, Pl)<br>

G = CrossGisherMatrix(El, CAB)<br>
V = CovAB(invW, GAB)<br>

yl = yQuadEstimator(dA, dB, El)<br>
cl = ClQuadEstimator(invW, yl)<br>
