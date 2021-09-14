#
# Implementation: python setup.py build_ext --inplace
# Generates PyCuAmpcor.xxx.so (where xxx is just some local sys-arch information).
# Note you need to run your makefile *FIRST* to generate the cuAmpcor.o object.
#

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

setup(  name = 'PyCuCohcor',
        version = '0.0.1',
        ext_modules = cythonize(Extension(
        "PyCuCohcor",
        sources=['PyCuCohcor.pyx'],
        include_dirs=['/usr/local/cuda/include', numpy.get_include()], # REPLACE WITH YOUR PATH TO YOUR CUDA LIBRARY HEADERS
        extra_compile_args=['-fPIC','-fpermissive'],
        extra_objects=['GDALImage.o','cuAmpcorChunk.o','cuAmpcorParameter.o','cuCorrFrequency.o',
                       'cuCorrNormalization.o','cuCorrNormalizationSAT.o', 'cuCorrNormalizer.o',
                       'cuCorrTimeDomain.o','cuArraysCopy.o','cuWindowFilter.o',
                       'cuArrays.o','cuArraysPadding.o','cuOffset.o','cuOverSampler.o',
                       'cuSincOverSampler.o', 'cuDeramp.o','cuAmpcorController.o','cuEstimateStats.o'],
        extra_link_args=['-L/usr/local/cuda/lib64',
                        '-L/usr/lib64/nvidia', '-L/opt/anaconda3/envs/isce2/lib',
                        '-lcudart','-lcufft','-lgdal'], # REPLACE FIRST PATH WITH YOUR PATH TO YOUR CUDA LIBRARIES
        language='c++'
    )))
