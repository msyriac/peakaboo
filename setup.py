#from setuptools import setup
from distutils.core import setup, Extension
import os



setup(name='peakaboo',
      version='0.1',
      description='Tools for Gravitational Lensing Peaks',
      url='https://github.com/msyriac/peakaboo',
      author='Mathew Madhavacheril',
      author_email='mathewsyriac@gmail.com',
      license='BSD-2-Clause',
      packages=['peakaboo'],
      zip_safe=False)
