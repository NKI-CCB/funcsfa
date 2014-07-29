from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize

setup(
    name='sfa',
    version='0.1-dev',
    author='Tycho Bismeijer',
    author_email='t.bismeijer@nki.nl',
    packages=['sfa'],
    ext_modules=cythonize([Extension('sfa._sfa',
                                     ['sfa/_sfa.pyx', 'sfa-c/src/sfa.c'],
                                     libraries=['blas', 'lapacke'],
                                     include_dirs=['sfa-c/include'])]),
)
