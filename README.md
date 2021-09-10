# Automatic liver segmentation in CT using deep learning
[![license](https://img.shields.io/github/license/DAVFoundation/captain-n3m0.svg?style=flat-square)](https://github.com/DAVFoundation/captain-n3m0/blob/master/LICENSE)

#### Trained U-Net on the LITS dataset is automatically downloaded when running the inference script and can be used as you wish, ENJOY! :)

<img src="figures/Segmentation_CustusX.PNG" width="70%" height="70%">

The figure shows a predicted liver mask with the corresponding patient CT in 3DSlicer. It is the Volume-10 from the LITS17 dataset.

## Install

```
pip install git+https://github.com/andreped/livermask.git
```

## Usage:

```
livermask --input path-to-nifti.nii --output path-to-output-file.nii
```

In addition, there is the optional `--cpu` action to disable the GPU (force computations on CPU only) if necessary.

## DICOM/NIfTI format
Pipeline assumes input is in the NIfTI format, and output a binary volume in the same format (.nii).
DICOM can be converted to NIfTI using the CLI [dcm2niix](https://github.com/rordenlab/dcm2niix), as such:
```
dcm2niix -s y -m y -d 1 "path_to_CT_folder" "output_name"
```

Note that "-d 1" assumed that "path_to_CT_folder" is the folder just before the set of DICOM scans you want to import and convert. This can be removed if you want to convert multiple ones at the same time. It is possible to set "." for "output_name", which in theory should output a file with the same name as the DICOM folder, but that doesn't seem to happen...

## Troubleshooting
You might have issues downloading the model when using VPN. If any issues are observed, try to disable VPN and try again.

## Acknowledgements
The LITS dataset can be accessible from [here](https://competitions.codalab.org), and the corresponding paper for the challenge from [here](https://arxiv.org/abs/1901.04056). If trained model is used, please consider citing this paper.

------

Made with :heart: and python
