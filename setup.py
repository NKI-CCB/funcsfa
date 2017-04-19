from setuptools import setup
from setuptools.extension import Extension

import numpy
from numpy.distutils.system_info import get_info


setup(
    name='sfamd',
    version='1.0.dev17',
    description='Sparse Factor Analysis of Multiple Datatypes',
    author='Tycho Bismeijer',
    author_email='t.bismeijer@nki.nl',
    url='http://ccb.nki.nl/software/sfamd/',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3.5'
    ],
    ext_modules=[Extension(
        'sfamd._sfamd',
        ['sfamd/_sfamd.pyx', 'sfa-c/src/sfamd.c', 'sfa-c/build/version.c'],
        libraries=(get_info('blas_opt')['libraries'] +
                   get_info('lapack_opt')['libraries'] +
                   ['m']),
        include_dirs=['sfa-c/include', numpy.get_include()])],
    packages=['sfamd']
)
