from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

_VERSION = '0.3'

REQUIRED_PACKAGES = [
]

DEPENDENCY_LINKS = [
]

setuptools.setup(
    name='warmup_scheduler',
    version=_VERSION,
    description='Gradually Warm-up LR Scheduler for Pytorch',
    install_requires=REQUIRED_PACKAGES,
    dependency_links=DEPENDENCY_LINKS,
    url='https://github.com/ildoonet/pytorch-gradual-warmup-lr',
    license='MIT License',
    package_dir={},
    packages=setuptools.find_packages(exclude=['tests']),
)
