# Automatic liver segmentation in CT using deep learning

#### NOTE: Trained 2D model on the LITS dataset is automatically downloaded when running the inference script and can be used as you wish, but please, give credit. ENJOY! :)


![Screenshot](figures/Segmentation_CustusX.PNG)

First of all:
The LITS dataset can be accessible from here (https://competitions.codalab.org), and the corresponding paper for the challenge (Bilic. P et al.. (2019). The Liver Tumor Segmentation Benchmark (LiTS). https://arxiv.org/abs/1901.04056).

Usage:
1) Simply clone the git
2) Create and activate virtual environment with dependencies (run > python setup.py for dependencies)
3) The inference script is located in the lungmask subfolder. Deploy the trained model on a user-specified CT running:
> python lungmask.py *path_to_nifti_file.nii* *output_name.nii*

NOTE: Currently, model only works for the nifti format, and outputs a binary volume in the same format (*.nii). But this format can be imported in CustusX. I wouldn't recommend mixing DICOM and .nii prediction file in CustusX, as there seem to be some orientation issues between these (bug to be fixed in the future). But simply convert DICOM -> NIFTI using the command-line tool dcm2niix (https://github.com/rordenlab/dcm2niix).

Convert DICOM -> NIFTI doing this:
> dcm2niix -s y -m y "path_to_CT_folder" "output_name"

It is possible to set "." for "output_name", which in theory should output a file with the same name as the DICOM folder, but that doesn't seem to happen...

A few final notes:
1) Requirements.txt is outdated. I would use setup.py for installing the correct dependencies. Will update requirements.txt in the future.
2) If you get SSLError during downloading the model, disable VPN, e.g. cisco. For those on the sintef network, try changing network to Eduroam or similar, as it might be a most-famous evry-issue...
