"""
Set of routines to ...

Author: Vanneste
"""
from __future__ import division

import numpy as np

def polrotangle(ri, rj):
    """
    ???

    Parameters
    ----------
    ri : ???
        ???
    rj : ???
        ???

    Returns
    ----------
    cos2a : 1D array of floats
        ???
    sin2a : 1D array of floats
        ???
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
    if norm<=1e-15:
        cos2a = 1.0
        sin2a = 0.0
        return cos2a, sin2a
    ris = ris / norm

    # Now, the angle we want is that between these two great circles: defined by
    cosa = np.dot(rij, np.transpose(ris))

    # the sign is more subtle : see tegmark et de oliveira costa 2000 eq. A6
    rijris = np.cross(rij, ris)
    sina = np.dot(rijris, np.transpose(ri))

    # so now we have directly cos2a and sin2a
    cos2a = 2.0 * cosa * cosa - 1.0
    sin2a = 2.0 * cosa * sina

    return cos2a, sin2a
