---
title: 'livermask: Automatic Liver Parenchyma and vessel segmentation in CT'
colorFrom: indigo
colorTo: indigo
sdk: docker
app_port: 7860
emoji: 🔎
pinned: false
license: mit
app_file: demo/app.py
---


<div align="center">
<h1 align="center">livermask</h1>
<h3 align="center">Automatic liver parenchyma and vessel segmentation in CT using deep learning</h3>

[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/DAVFoundation/captain-n3m0/blob/master/LICENSE)
[![Build Actions Status](https://github.com/andreped/livermask/workflows/Build/badge.svg)](https://github.com/andreped/livermask/actions)
[![DOI](https://zenodo.org/badge/238680374.svg)](https://zenodo.org/badge/latestdoi/238680374)
[![GitHub Downloads](https://img.shields.io/github/downloads/andreped/livermask/total?label=GitHub%20downloads&logo=github)](https://github.com/andreped/livermask/releases)
[![Pip Downloads](https://img.shields.io/pypi/dm/livermask?label=pip%20downloads&logo=python)](https://pypi.org/project/livermask/)
 
**livermask** was developed by SINTEF Medical Technology to provide an open tool to accelerate research.

<img src="figures/Segmentation_3DSlicer.PNG" width="70%">
</div>

## Demo <a target="_blank" href="https://huggingface.co/spaces/andreped/livermask"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg"></a>

An online version of the tool has been made openly available at Hugging Face spaces, to enable researchers to easily test the software on their own data without downloading it. To access it, click on the badge above.

## Install

A stable release is available on PyPI:
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

Program has been tested using Python 3.7 on Windows, macOS, and Ubuntu Linux 20.04.

## Usage

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
| `--extension` | which extension to save output in (default: `.nii`) |

<details open>
<summary>

### Using code directly</summary>
If you wish to use the code directly (not as a CLI and without installing), you can run this command:
```
python -m livermask.livermask --input path-to-input --output path-to-output
```
</details>

<details>
<summary>

### DICOM/NIfTI format</summary>
Pipeline assumes input is in the NIfTI format, and output a binary volume in the same format (.nii or .nii.gz).
DICOM can be converted to NIfTI using the CLI [dcm2niix](https://github.com/rordenlab/dcm2niix), as such:
```
dcm2niix -s y -m y -d 1 "path_to_CT_folder" "output_name"
```

Note that "-d 1" assumed that "path_to_CT_folder" is the folder just before the set of DICOM scans you want to import and convert. This can be removed if you want to convert multiple ones at the same time. It is possible to set "." for "output_name", which in theory should output a file with the same name as the DICOM folder, but that doesn't seem to happen...

</details>


<details>
<summary>

### Troubleshooting</summary>
You might have issues downloading the model when using VPN. If any issues are observed, try to disable VPN and try again.

If the program struggles to install, attempt to install using:
```
pip install --force-reinstall --no-deps git+https://github.com/andreped/livermask.git
```

If you experience issues with numpy after installing CuPy, try reinstalling CuPy with this extension:
```
pip install 'cupy-cuda110>=7.7.0,<8.0.0'
```
</details>

## Applications of livermask
* Yevdokimov et al., Recognition of Diffuse Hepatic Steatosis, 33rd Conference of Open Innovations Association (FRUCT), 2023, https://doi.org/10.23919/FRUCT58615.2023.10143062
* Pérez de Frutos et al., Learning deep abdominal CT registration through adaptive loss weighting and synthetic data generation, PLOS ONE, 2023, 
https://doi.org/10.1371/journal.pone.0282110
* Lee et al., Robust End-to-End Focal Liver Lesion Detection Using Unregistered Multiphase Computed Tomography Images, IEEE Transactions on Emerging Topics in Computational Intelligence, 2021, https://doi.org/10.1109/TETCI.2021.3132382
* Survarachakan et al., Effects of Enhancement on Deep Learning Based Hepatic Vessel Segmentation, Electronics, 2021, https://doi.org/10.3390/electronics10101165

## Segmentation performance metrics
The segmentation models were evaluated on an internal dataset against manual annotations. See Table E in S4 Appendix in the Supporting Information of [this paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282110) for more information. The table presented there can also be seen below:

| Class | DSC | HD95 |
|--------|-------------------|------------------|
| Parenchyma | 0.946±0.046 | 10.122±11.032 |
| Vessels | 0.355±0.090 | 24.872±5.161 |

The parenchyma segmentation model was trained on the LITS dataset, whereas the vessel model was trained on a local dataset. The LITS dataset is openly accessible and can be downloaded from [here](https://competitions.codalab.org/competitions/17094).

## Acknowledgements
If you found this tool helpful in your research, please, consider citing it (see [here](https://zenodo.org/badge/latestdoi/238680374) for more information on how to cite):
<pre>
@software{andre_pedersen_2023_7574587,
  author       = {André Pedersen and Javier Pérez de Frutos},
  title        = {andreped/livermask: v1.4.1},
  month        = jan,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v1.4.1},
  doi          = {10.5281/zenodo.7574587},
  url          = {https://doi.org/10.5281/zenodo.7574587}
}
</pre>

In addition, the segmentation performance of the tool was presented in [this paper](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0282110), thus, cite this tool as well if that is of relevance for you study:
<pre>
@article{perezdefrutos2022ddmr,
    title = {Learning deep abdominal CT registration through adaptive loss weighting and synthetic data generation},
    author = {Pérez de Frutos, Javier AND Pedersen, André AND Pelanis, Egidijus AND Bouget, David AND Survarachakan, Shanmugapriya AND Langø, Thomas AND Elle, Ole-Jakob AND Lindseth, Frank},
    journal = {PLOS ONE},
    publisher = {Public Library of Science},
    year = {2023},
    month = {02},
    volume = {18},
    doi = {10.1371/journal.pone.0282110},
    url = {https://doi.org/10.1371/journal.pone.0282110},
    pages = {1-14},
    number = {2}
}
</pre>
