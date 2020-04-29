#!/usr/bin/env python

from setuptools import setup

dependencies = ["segmentation-models"]

setup(name='poc-detection',
      version='0.0.1',
      description='POC detection model',
      license='GPLv3',
      author='Duncan Watson-Parris',
      author_email='duncan.watson-parris@physics.ox.ac.uk',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'Topic :: Scientific/Engineering :: Physics',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.7',
      ],
      keywords=['climate', 'machine-learning'],
      install_requires=dependencies,
      )