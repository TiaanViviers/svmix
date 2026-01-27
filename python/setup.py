"""
Setup script for svmix Python package.

Note: pyproject.toml is the primary configuration.
This file is kept for backward compatibility.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_file = Path(__file__).parent.parent / "README.md"
if readme_file.exists():
    long_description = readme_file.read_text()
else:
    long_description = "Stochastic Volatility Model Mixture for Bayesian Filtering"

setup(
    name="svmix",
    version="1.0.0",
    author="Tiaan Viviers",
    description="Stochastic Volatility Model Mixture for Bayesian Filtering",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TiaanViviers/svmix",
    packages=find_packages(),
    package_data={
        'svmix': ['lib/*.so', 'lib/*.dylib', 'lib/*.dll'],
    },
    install_requires=[
        'numpy>=1.20.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov',
        ],
    },
    python_requires='>=3.8',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
        "Programming Language :: Python :: 3",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering",
    ],
)
