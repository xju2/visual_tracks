from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from setuptools import find_packages, setup

description = "Facility for accessing and visualizing tracking data"

setup(
    name="acctrack",
    version="0.1.0",
    description=description,
    author="Xiangyang Ju",
    author_email="xiangyang.ju@gmail.com",
    url="https://github.com/xju2/visual_tracks",
    license="MIT",
    packages=find_packages(),
    install_requires=[
    ],
    extras_require={},
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    scripts=[
        "scripts/split_files_for_nn.py",
    ],
)
