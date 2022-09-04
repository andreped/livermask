<div align="center">
<h1 align="center">livermask</h1>
<h3 align="center">Automatic liver parenchyma and vessel segmentation in CT using deep learning</h3>

[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/DAVFoundation/captain-n3m0/blob/master/LICENSE)
[![Build Actions Status](https://github.com/andreped/livermask/workflows/Build/badge.svg)](https://github.com/andreped/livermask/actions)
[![DOI](https://zenodo.org/badge/238680374.svg)](https://zenodo.org/badge/latestdoi/238680374)
[![GitHub Downloads](https://img.shields.io/github/downloads/andreped/livermask/total?label=GitHub%20downloads&logo=github)](https://github.com/andreped/livermask/releases)
[![Pip Downloads](https://img.shields.io/pypi/dm/livermask?label=pip%20downloads&logo=python)](https://pypi.org/project/livermask/)
 
**livermask** was developed by SINTEF Medical Technology to provide an open tool for liver parenchyma segmentation to accelerate research.

<img src="figures/Segmentation_3DSlicer.PNG" width="70%">
</div>

## Install

```
pip install livermask
```

Alternatively, to install from source do:
```
pip install git+https://github.com/andreped/livermask.git
```

As TensorFlow 2.4 only supports Python 3.6-3.8, so does livermask. Software 
is also compatible with Anaconda. However, best way of installing livermask is using `pip`, which
also works for conda environments.

(Optional) To add GPU inference support for liver vessel segmentation (which uses Chainer and CuPy), you need to install [CuPy](https://github.com/cupy/cupy). This can be easily done by adding `cupy-cudaX`, where `X` is the CUDA version you have installed, for instance `cupy-cuda110` for CUDA-11.0:
```
pip install cupy-cuda110
```

(DEBUG) If you experience issues with numpy, after installing CuPy, try reinstalling CuPy with this extension:
```
pip install 'cupy-cuda110>=7.7.0,<8.0.0'
```

Program has been tested using Python 3.7 on Windows, macOS, and Ubuntu Linux 18.04.

## Usage:

```
livermask --input path-to-input --output path-to-output
```

|  command<img width=10/> | description |
| ------------------- | ------------- |
| `--input`  | the full path to the input data. Could be nifti file or directory (if directory is provided as input) |
| `--output`  | the full path to the output data. Could be either output name or directory (if directory is provided as input)  |
| `--cpu`  | to disable the GPU (force computations on CPU only) |
| `--verbose`  | to enable verbose |
| `--vessels` | to segment vessels |

### Using code directly:
If you wish to use the code directly (not as a CLI and without installing), you can run this command:
```
python -m livermask.livermask --input path-to-input --output path-to-output
```

## DICOM/NIfTI format
Pipeline assumes input is in the NIfTI format, and output a binary volume in the same format (.nii).
DICOM can be converted to NIfTI using the CLI [dcm2niix](https://github.com/rordenlab/dcm2niix), as such:
```
dcm2niix -s y -m y -d 1 "path_to_CT_folder" "output_name"
```

Note that "-d 1" assumed that "path_to_CT_folder" is the folder just before the set of DICOM scans you want to import and convert. This can be removed if you want to convert multiple ones at the same time. It is possible to set "." for "output_name", which in theory should output a file with the same name as the DICOM folder, but that doesn't seem to happen...

## Troubleshooting
You might have issues downloading the model when using VPN. If any issues are observed, try to disable VPN and try again.

If the program struggles to install, attempt to install using:
```
pip install --force-reinstall --no-deps git+https://github.com/andreped/livermask.git
```

## Acknowledgements
If you found this tool helpful in your research, please, consider citing it:
<pre>
  @software{andre_pedersen_2021_5812222,
  author       = {André Pedersen},
  title        = {andreped/livermask: v1.3.1},
  month        = dec,
  year         = 2021,
  publisher    = {Zenodo},
  version      = {v1.3.1},
  doi          = {10.5281/zenodo.5812222},
  url          = {https://doi.org/10.5281/zenodo.5812222}}
</pre>

Information on how to cite can be found [here](https://zenodo.org/badge/latestdoi/238680374).

The model was trained on the LITS dataset. The dataset is openly accessible and can be downloaded from [here](https://competitions.codalab.org/competitions/17094).

------

Made with :heart: and python
