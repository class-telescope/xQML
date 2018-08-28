"""
Set of routines to ...

Author: Vanneste
"""
from __future__ import division

import numpy as np


def polrotangle(ri, rj):
    """
    Computes cosine and sine of twice the angle between pixels i and j.

    Parameters
    ----------
    ri : 3D array of floats
        Coordinates of vector corresponding to input pixels i following
        healpy.pix2vec(nside,ipix) output
    rj : 3D array of floats
        Coordinates of vector corresponding to input pixels j following
        healpy.pix2vec(nside,jpix) output

    Returns
    ----------
    cos2a : 1D array of floats
        Cosine of twice the angle between pixels i and j
    sin2a : 1D array of floats
        Sine of twice the angle between pixels i and j

    Example
    ----------
    >>> cos2a, sin2a = polrotangle([0.1,0.2,0.3], [0.4,0.5,0.6])
    >>> print(round(cos2a,5),round(sin2a,5))
    (0.06667, 0.37333)
    """
    z = np.array([0.0, 0.0, 1.0])

    # Compute ri^rj : unit vector for the great circle connecting i and j
    rij = np.cross(ri, rj)
    norm = np.sqrt(np.dot(rij, np.transpose(rij)))

    # case where pixels are identical or diametrically opposed on the sky
    if norm <= 1e-15:
        cos2a = 1.0
        sin2a = 0.0
        return cos2a, sin2a
    rij = rij / norm

    # Compute z^ri : unit vector for the meridian passing through pixel i
    ris = np.cross(z, ri)
    norm = np.sqrt(np.dot(ris, np.transpose(ris)))

    # case where pixels is at the pole
    if norm <= 1e-15:
        cos2a = 1.0
        sin2a = 0.0
        return cos2a, sin2a
    ris = ris / norm

    # Now, the angle we want is that
    # between these two great circles: defined by
    cosa = np.dot(rij, np.transpose(ris))

    # the sign is more subtle : see tegmark et de oliveira costa 2000 eq. A6
    rijris = np.cross(rij, ris)
    sina = np.dot(rijris, np.transpose(ri))

    # so now we have directly cos2a and sin2a
    cos2a = 2.0 * cosa * cosa - 1.0
    sin2a = 2.0 * cosa * sina

    return cos2a, sin2a

if __name__ == "__main__":
    """
    Run the doctest using

    python simulation.py

    If the tests are OK, the script should exit gracefuly, otherwise the
    failure(s) will be printed out.
    """
    import doctest
    if np.__version__ >= "1.14.0":
        np.set_printoptions(legacy="1.13")
    doctest.testmod()
