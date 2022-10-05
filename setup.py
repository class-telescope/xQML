#!/usr/bin/env python
import sys
import numpy as np
import setuptools
from distutils.core import setup, Extension
#from numpy.distutils.core import setup
#from numpy.distutils.extension import Extension
from Cython.Distutils import build_ext

# Get numpy include directory (works across versions)
try:
    numpy_include = np.get_include()
except AttributeError:
    numpy_include = np.get_numpy_include()

print(sys.argv)

if '--disable-openmp' in sys.argv:
    sys.argv.pop(sys.argv.index('--disable-openmp'))
    USE_OPENMP = False
else:
    USE_OPENMP = True

if '--icc' in sys.argv:
    sys.argv.pop(sys.argv.index('--icc'))
    USE_ICC = True
else:
    USE_ICC = False

libs = ['openblas', 'm']

extra = ['-std=gnu99']
if USE_ICC:
    if USE_OPENMP:
        libs += ['gomp', 'iomp5']
        extra += ['-openmp']
else:
   # extra += ['-O2']
    if USE_OPENMP:
        libs += ['gomp']
        extra += ['-fopenmp']


libcov = Extension(name="xqml._libcov",
                   sources=["src/libcov.c","src/_libcov.pyx"],
                   libraries=libs,
                   include_dirs=['src', numpy_include],
                   library_dirs=["src"],
                   extra_compile_args=extra
                   )

setup(name="xqml",
      description="Code for xQML",
      author="S. Vanneste & M. Tristram",
      version="0.2",
      packages=['xqml'],
      ext_modules=[libcov],
      install_requires=["numpy","scipy>=1.1.0","healpy>=0.6.1"],
      cmdclass={'build_ext': build_ext}
      )
