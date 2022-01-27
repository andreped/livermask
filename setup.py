from setuptools import setup, find_packages
from setuptools.command.install import install
import os


with open("README.md", "r") as f:
    long_description = f.read().decode("utf-16")

with open('requirements.txt', 'r') as ff:
    required = ff.read().decode("utf-16").splitlines()


setup(
     name='livermask',  
     version='1.3.1',
     author="André Pedersen",
     author_email="andrped94@gmail.com",
     license='MIT',
     description="A package for automatic segmentation of liver from CT data",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/andreped/livermask",
     include_package_data=True,
     packages=find_packages(
        include=[
            'livermask', 
            'livermask.utils',
            'livermask.configs',
        ]
    ),
     entry_points={
        'console_scripts': [
            'livermask = livermask.livermask:main',
        ]
     },
     install_requires=required,
     classifiers=[
         "Programming Language :: Python :: 3.6",
         "Programming Language :: Python :: 3.7",
         "Programming Language :: Python :: 3.8",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     python_requires='>=3.6',
 )
