"""Setup script for autodiff."""

from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = ['numpy']

setup(
    name='autodiff',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    include_package_data=True,
    packages=[p for p in find_packages() if p.startswith('autodiff')],
    description='Simply auto differentiation package',
)