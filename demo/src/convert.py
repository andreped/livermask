import nibabel as nib
from nibabel.processing import resample_to_output
from skimage.measure import marching_cubes


def nifti_to_glb(path, output="prediction.obj"):
    # load NIFTI into numpy array
    image = nib.load(path)
    resampled = resample_to_output(image, [1, 1, 1], order=1)
    data = resampled.get_fdata().astype("uint8")

    # extract surface
    verts, faces, normals, values = marching_cubes(data, 0)
    faces += 1

    with open(output, 'w') as thefile:
        for item in verts:
            thefile.write("v {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in normals:
            thefile.write("vn {0} {1} {2}\n".format(item[0],item[1],item[2]))

        for item in faces:
            thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0],item[1],item[2]))
