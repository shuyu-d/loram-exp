from setuptools import setup
from Cython.Build import cythonize

setup(
    keywords='structure learning causal discovery bayesian network ',
    install_requires=['numpy', 'scipy', 'python-igraph'],
    ext_modules = cythonize(["spmaskmatmul.pyx"])
)

