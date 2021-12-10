from setuptools import setup, find_packages
from setuptools.command.install import install
import os


with open("README.md", "r") as f:
    long_description = f.read()

with open('requirements.txt', 'r', encoding='utf-16') as ff:
    required = ff.read().splitlines()


class InstallCommand(install):
    user_option = install.user_options + [
        ('cupy=', 'cupy', 'enable flag to install package with GPU support'),
    ]

    def initialize_options(self):
        install.initialize_options(self)
        self.cupy = "cupy"

    def finalize_options(self):
        print("CuPY version selected is: ", self.cupy)
        install.finalize_options(self)

    def run(self): 
        required.append(self.cupy)
        install.run(self)


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
     packages=find_packages("livermask"),
     entry_points={
        'console_scripts': [
            'livermask = livermask.livermask:main',
            'unet3d = livermask.unet3d:UNet3D',
            #'utils = livermask.utils',
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
     cmdclass={'install': InstallCommand},
     dependency_links=[
        os.path.join(os.getcwd(), 'deps', 'my_package-1.0.0-py3.5.egg')
     ]
 )
