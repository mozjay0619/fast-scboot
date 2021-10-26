#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os
import pprint
import sys

from setuptools import find_packages, setup
from setuptools.command.install import install
from setuptools.extension import Extension

# Set up the logging environment
logging.basicConfig()
log = logging.getLogger()

# Use Cython if available
try:
    from Cython.Distutils import build_ext

    # Use Cython’s build_ext module which runs cythonize as part of the build process
except ImportError:
    USE_CYTHON = False
else:
    USE_CYTHON = True

cmdclass = {}
ext_modules = []

# Extension options (numpy header file)
include_dirs = []
try:
    import numpy as np

    include_dirs.append(np.get_include())
except ImportError:
    log.critical("Numpy and its headers are required to run setup(). Exiting")
    sys.exit(1)

opts = dict(
    include_dirs=include_dirs,
)
log.debug("opts:\n%s", pprint.pformat(opts))

# Build extension modules
if USE_CYTHON:
    ext_modules += [
        Extension(
            "fast_scboot.c.tuple_hash_function",
            ["src/fast_scboot/c/tuple_hash_function.pyx"],
            **opts
        ),
        Extension("fast_scboot.c.utils", ["src/fast_scboot/c/utils.pyx"], **opts),
        Extension(
            "fast_scboot.c.sample_index_helper",
            ["src/fast_scboot/c/sample_index_helper.pyx"],
            **opts
        ),
        Extension(
            "fast_scboot.c.pytest_validator",
            ["src/fast_scboot/c/pytest_validator.pyx"],
            **opts
        ),
    ]
    # First argument is the compilation target location.
    cmdclass.update({"build_ext": build_ext})

else:
    ext_modules += [
        Extension(
            "fast_scboot.c.tuple_hash_function",
            ["src/fast_scboot/c/tuple_hash_function.c"],
            **opts
        ),
        Extension("fast_scboot.c.utils", ["src/fast_scboot/c/utils.c"], **opts),
        Extension(
            "fast_scboot.c.sample_index_helper",
            ["src/fast_scboot/c/sample_index_helper.c"],
            **opts
        ),
        Extension(
            "fast_scboot.c.pytest_validator",
            ["src/fast_scboot/c/pytest_validator.c"],
            **opts
        ),
    ]

# circleci.py version
VERSION = "v0.1.b2"

# circleci version verfication
class VerifyVersionCommand(install):
    """Custom command to verify that the git tag matches our version"""

    description = "verify that the git tag matches our version"

    def run(self):
        tag = os.getenv("CIRCLE_TAG")

        if tag != VERSION:
            info = "Git tag: {0} does not match the version of this app: {1}".format(
                tag, VERSION
            )
            sys.exit(info)


cmdclass.update({"verify": VerifyVersionCommand})


def readme():
    """print long description"""
    with open("README.rst") as f:
        return f.read()


setup(
    name="fast-scboot",
    version=VERSION,
    description="Fast implementation of the stratified cluster bootstrap sampling algorithm.",
    long_description=readme(),
    url="https://github.com/mozjay0619/fast-scboot",
    author="Jay Kim",
    author_email="mozjay0619@gmail.com",
    license="DSB 3-clause",
    cmdclass=cmdclass,
    ext_modules=ext_modules,
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"": ["*.pxd", "*.pyx"]},
    python_requires=">=3",
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

# python3 setup.py build_ext --inplace
