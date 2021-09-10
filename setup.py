import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

with open('misc/requirements_torch.txt', 'r', encoding='utf-16') as ff:
    required = ff.read().splitlines()

setuptools.setup(
     name='livermask',  
     version='1.0.0',
     author="André Pedersen",
     author_email="andrped94@gmail.com",
     license='MIT',
     description="A package for automatic segmentation of liver from CT data",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/andreped/livermask",
     packages=setuptools.find_packages(),
     entry_points={
        'console_scripts': [
            'livermask = livermask.livermask:main'
        ]
     },
     install_requires=required,
     classifiers=[
         "Programming Language :: Python :: 3",
         "License :: OSI Approved :: MIT License",
         "Operating System :: OS Independent",
     ],
     python_requires='>=3.6',
 )
