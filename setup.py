#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 16:11:18 2020

@author: ebecerra
"""

from setuptools import setup

setup(name='gym_forest_fire',
      version='2.4',
      install_requires=['gym',
                        'numpy',
                        'matplotlib',
                        'seaborn',
                        'cython',
                        'imageio',
                        'tdqm',
                        'math',
                        ]
)
