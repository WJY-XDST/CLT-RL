# Copyright (c) 2026, Wheel-Legged-Gym Isaac Lab Migration
# SPDX-License-Identifier: BSD-3-Clause

"""Installation script for the 'wheel_legged_gym_isaaclab' package."""

import os
import toml

from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
EXTENSION_TOML_DATA = toml.load(
    os.path.join(EXTENSION_PATH, "config", "extension.toml")
)

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # Isaac Sim 5.1 pins this package.  Leaving it unconstrained lets a
    # development-mode install upgrade it and break Isaac Sim at import time.
    "psutil==5.9.8",
]

setup(
    name=EXTENSION_TOML_DATA["package"]["name"],
    packages=find_packages(),
    package_data={
        "": ["*.usd", "*.urdf", "*.STL", "*.stl", "*.csv", "*.yaml", "*.yml"],
        "wheel_legged_gym_isaaclab": [
            "assets/**/*",
            "assets/robots/wl/urdf/*",
            "assets/robots/wl/meshes/*",
        ],
    },
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="BSD-3-Clause",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 5.1.0",
    ],
    zip_safe=False,
)
