from setuptools import setup, find_packages
from setuptools.command.install import install
import os, sys


print("\n\n\n\n---:", sys.args)


with open("README.md", "r") as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-16') as ff:
    required = ff.read().splitlines()


class InstallCommand(install):
    user_option = install.user_options + [
        ('cupyyy=', None, 'enable flag to install package with GPU support'),
    ]

    def initialize_options(self):
        super().initialize_options()
        self.cupy = "cupy"

    def finalize_options(self):
        print("CuPy version selected is: ", self.cupyyy)
        super().finalize_options()

    def run(self):
        # use options
        global cupy
        required.append(self.cupy)

        super().run()


setup(
     name='livermask',  
     version='1.2.0',
     author="André Pedersen",
     author_email="andrped94@gmail.com",
     license='MIT',
     description="A package for automatic segmentation of liver from CT data",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/andreped/livermask",
     packages=find_packages(
        include=[
            'livermask', 
            'livermask.utils',
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
         "Programming Language :: Python :: 3.9",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     python_requires='>=3.6',
     cmdclass={
        'install': InstallCommand,
     },
 )
