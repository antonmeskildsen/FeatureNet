#!/usr/bin/env python

from setuptools import setup

setup(name='FeatureNet',
      version='0.1',
      description='Neural network powered eye feature detection system',
      author='Anton Mølbjerg Eskildsen',
      author_email='antonmeskildsen@me.com',
      packages=['featurenet'],
      setup_requires=[
          'torch',
          'torchvision',
          'torchsample',
          'torchnet'
      ],
      dependency_links=[
          'https://github.com/ncullen93/torchsample/tarball/master#egg=torchsample',
          'https://github.com/pytorch/tnt/tarball/master#egg=torchnet'
      ])
