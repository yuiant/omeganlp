#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs
import os.path

from setuptools import find_packages, setup


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="omeganlp",
    version=get_version('omeganlp/__init__.py'),
    description="NLP Framework based on pytorch",
    author="yuiant",
    author_email="gyuiant@gmail.com",
    url="https://github.com/yuiant/omeganlp.git",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[],
    test_suite="test",
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"]
)
