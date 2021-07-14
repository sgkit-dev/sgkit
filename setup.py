#!/usr/bin/env python
from setuptools import setup

setup(
    # The package name along with all the other metadata is specified in setup.cfg
    # However, GitHub's dependency graph can't see the package unless we put this here.
    name="sgkit",
    use_scm_version=True,
)
