from setuptools import setup
from setuptools.extension import Extension

import numpy
from numpy.distutils.system_info import get_info


setup(
    name='funcsfa',
    version='1.0.dev22',
    description='Functional Sparse Factor Analysis of Multiple Datatypes',
    author='Tycho Bismeijer',
    author_email='t.bismeijer@nki.nl',
    url='https://github.com/NKI-CCB/funcsfa',
    classifiers=[
        'Development Status :: 3 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: Other/Proprietary License',
        'Programming Language :: Python :: 3.5'
    ],
    ext_modules=[Extension(
        'funcsfa._lib',
        ['funcsfa/_lib.pyx', 'funcsfa-c/src/funcsfa.c',
         'funcsfa-c/build/version.c'],
        libraries=(get_info('blas_opt')['libraries'] +
                   get_info('lapack_opt')['libraries'] +
                   ['m']),
        include_dirs=['funcsfa-c/include', numpy.get_include()])],
    packages=['funcsfa']
)
