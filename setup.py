from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

from numpy.distutils.system_info import get_info


setup(
    name='sfamd',
    version='1.0.dev8',
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
        'Programming Language :: Python :: 3.4'
    ],
    ext_modules=cythonize([Extension(
        'sfamd._sfamd',
        ['sfamd/_sfamd.pyx', 'sfa-c/src/sfamd.c', 'sfa-c/build/version.c'],
        libraries=(get_info('blas_opt')['libraries'] +
                   get_info('lapack_opt')['libraries'] +
                   ['lapacke', 'm']),
        include_dirs=['sfa-c/include'])]),
    packages=['sfamd']
)
