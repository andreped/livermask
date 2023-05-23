import gradio as gr
import subprocess as sp
from skimage.measure import marching_cubes
import nibabel as nib
from nibabel.processing import resample_to_output


def nifti_to_glb(path):
    # load NIFTI into numpy array
    image = nib.load(path)
    resampled = resample_to_output(image, [1, 1, 1], order=1)
    data = resampled.get_fdata().astype("uint8")

    # extract surface
    verts, faces, normals, values = marching_cubes(data, 0)
    faces += 1

    with open('prediction.obj', 'w') as thefile:
        for item in verts:
            thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in normals:
            thefile.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in faces:
            thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))


def run_model(input_path):
    sp.check_call(["livermask", "--input", input_path, "--output", "prediction", "--verbose"])


def load_mesh(mesh_file_name):
    path = mesh_file_name.name
    run_model(path)
    nifti_to_glb("prediction-livermask.nii")
    return "./prediction.obj"


if __name__ == "__main__":
    demo = gr.Interface(
        fn=load_mesh,
        inputs=gr.UploadButton(label="Click to Upload a File", file_types=[".nii", ".nii.nz"], file_count="single"),
        outputs=gr.Model3D(clear_color=[0.0, 0.0, 0.0, 0.0], label="3D Model"),
        title="livermask: Automatic Liver Parenchyma segmentation in CT",
        description="Using pretrained deep learning model trained on the LiTS17 dataset",
    )
    demo.launch()
