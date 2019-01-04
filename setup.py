#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='opyt',
    version='0.0.dev0',
    author='Sam Brockie',
    author_email='sgb39@cam.ac.uk',
    packages=find_packages(),
    url='https://github.com/brocksam/opyt',
    license='All Rights Reserved',
    description=('Tools for optimizing dynamic systems using direct '
                 'collocation.'),
    long_description=open('README.rst').read(),
    python_requires='>=3.6',
    install_requires=[numpy>=1.15,
        scipy>=1.2,
        sympy>=1.3,
        cython>=0.29.2,
        cyipopt>=0.1.9],
    extras_require={'docs': [sphinx>=1.8]},
    tests_require=[pytest>=4.0])