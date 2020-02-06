# Automatic liver segmentation in CT using deep learning

Steps:
1) Simply clone the git
2) Create and activate virtual environment with dependencies (see requirements.txt)
3) The inference script is located in the lungmask subfolder. Deploy the model running:
> python lungmask.py *path_to_nifti_file.nii* *output_name.nii*

NOTE: Currently, model only works for the nifti format, and outputs a binary volume with extension *.nii. But this format can be imported in CustusX. I wouldn't recommend merging DICOM and .nii prediction file in CustusX, as there seem to be some orientation issues between these. But simply convert DICOM -> NIFTI using dcm2niix command-line tool (https://github.com/rordenlab/dcm2niix).

Convert DICOM -> NIFTI doing this:
> dcm2niix -s y -m y "path_to_CT_folder" .
