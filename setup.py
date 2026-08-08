from setuptools import setup, find_packages
import os

with open("README.md", "r", errors='ignore') as f:
    long_description = f.read()

setup(
    name='livermask',
    version='1.5.0',
    author="André Pedersen and Javier Pérez de Frutos",
    author_email="andrped94@gmail.com",
    license='MIT',
    description="A package for automatic segmentation of liver from CT data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/andreped/livermask",
    include_package_data=True,
    packages=find_packages(exclude=('figures')), 
    entry_points={
        'console_scripts': [
            'livermask = livermask.livermask:main',
        ]
    },
    install_requires=[
        "tensorflow>=2.4",
        "nibabel>=3.2",
        "scipy>=1.7",
        "scikit-image>=0.18",
        "chainer>=7.8",
        "gdown>=4.4",
        "tqdm>=4.62",
        "numba>=0.53",
        "requests>=2.26",
        "typing-extensions>=3.10",
        "importlib-metadata>=4.8.1",
        "Werkzeug>=2.0.1",
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
