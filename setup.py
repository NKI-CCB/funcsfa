from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

from numpy.distutils.system_info import get_info

setup(
    name='sfa',
    version='0.2-dev',
    author='Tycho Bismeijer',
    author_email='t.bismeijer@nki.nl',
    packages=['sfa'],
    ext_modules=cythonize([Extension(
        'sfa._sfa',
        ['sfa/_sfa.pyx', 'sfa-c/src/sfa.c'],
        libraries=get_info('blas_opt')['libraries'] +
                  get_info('lapack_opt')['libraries'] +
                  ['lapacke', 'm'],
        include_dirs=['sfa-c/include'])]),
)
