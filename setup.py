#!/usr/bin/env python
import sys
#import hooks
from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

# Get numpy include directory (works across versions)
try:
    numpy_include = numpy.get_include()
except AttributeError:
    numpy_include = numpy.get_numpy_include()

print( sys.argv)

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

libs = ['m']

extra = ['-std=gnu99']
if USE_ICC:
    if USE_OPENMP:
        libs += ['gomp', 'iomp5']
        extra += ['-openmp']
else:
    extra += ['-O4']
    if USE_OPENMP:
        libs += ['gomp']
        extra += ['-fopenmp']


libcov = Extension( name="_libcov",
                    sources=["src/libcov.c","src/_libcov.pyx"],
                    libraries=libs,
                    include_dirs=['src', numpy_include],
                    library_dirs=["src"],
                    extra_compile_args=extra
                    )



setup(name="xqml",
      description="Code for xQML",
      author="S. Vanneste & M. Tristram",
      version="0.1",
      packages=['xqml'],
      ext_modules=[libcov],
      cmdclass={'build_ext': build_ext}
      )
