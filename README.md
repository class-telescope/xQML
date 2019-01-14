xQML

a quadratic Power Spectrum estimator based on cross-correlation between maps.
[Vanneste et al., 2018, [astro-ph/1807.02484] (https://arxiv.org/abs/1807.02484) ]

This is the generalisation to cross-correlation of QML methods
[Tegmark Physical Review D 55, 5895 (1997), Tegmark and de Oliveira-Costa, Physical Review D 64 (2001)]


<br>
<br>

The code is in python with some C routines.<br>
But still UNDER DEVELOPMENT !

To install:
> pip install . --prefix=$PATH

To use:
> import xqml

We are working to externalise CPU intensive routines and use OpenMP.

Librairies needed are :
- scipy
- numpy
- healpy


