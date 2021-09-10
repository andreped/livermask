import numpy as np 
import os, sys
from tqdm import tqdm 
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to
from scipy.ndimage import zoom
from tensorflow.python.keras.models import load_model
import gdown
from skimage.morphology import remove_small_holes, binary_dilation, binary_erosion, ball
from skimage.measure import label, regionprops
import warnings
import argparse
import pkg_resources
import tensorflow as tf


# due to this: https://github.com/tensorflow/tensorflow/issues/35029
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


# mute some warnings
warnings.filterwarnings('ignore', '.*output shape of zoom.*')


def intensity_normalization(volume, intensity_clipping_range):
    result = np.copy(volume)

    result[volume < intensity_clipping_range[0]] = intensity_clipping_range[0]
    result[volume > intensity_clipping_range[1]] = intensity_clipping_range[1]

    min_val = np.amin(result)
    max_val = np.amax(result)
    if (max_val - min_val) != 0:
        result = (result - min_val) / (max_val - min_val)

    return result

def post_process(pred):
    return pred

def get_model(output):
    url = "https://drive.google.com/uc?id=12or5Q79at2BtLgQ7IaglNGPFGRlEgEHc"
    # output = "./model.h5"
    md5 = "ef5a6dfb794b39bea03f5496a9a49d4d"
    gdown.cached_download(url, output, md5=md5) #, postprocess=gdown.extractall)

def func(path, output, cpu):

    cwd = "/".join(os.path.realpath(__file__).replace("\\", "/").split("/")[:-1]) + "/"

    print("Current cwd:", cwd)
    name = cwd + "model.h5"

    # get model
    get_model(name)

    # load model
    model = load_model(name, compile=False)

    print("preprocessing...")
    nib_volume = nib.load(path)
    new_spacing = [1., 1., 1.]
    resampled_volume = resample_to_output(nib_volume, new_spacing, order=1)
    data = resampled_volume.get_data().astype('float32')

    curr_shape = data.shape

    # resize to get (512, 512) output images
    img_size = 512
    data = zoom(data, [img_size / data.shape[0], img_size / data.shape[1], 1.0], order=1)

    # intensity normalization
    intensity_clipping_range = [-150, 250] # HU clipping limits (Pravdaray's configs)
    data = intensity_normalization(volume=data, intensity_clipping_range=intensity_clipping_range)

    # fix orientation
    data = np.rot90(data, k=1, axes=(0, 1))
    data = np.flip(data, axis=0)

    print("predicting...")
    # predict on data
    pred = np.zeros_like(data).astype(np.float32)
    for i in tqdm(range(data.shape[-1]), "pred: "):
        pred[..., i] = model.predict(np.expand_dims(np.expand_dims(np.expand_dims(data[..., i], axis=0), axis=-1), axis=0))[0, ..., 1]
    del data 

    # threshold
    pred = (pred >= 0.4).astype(int)

    # fix orientation back
    pred = np.flip(pred, axis=0)
    pred = np.rot90(pred, k=-1, axes=(0, 1))

    print("resize back...")
    # resize back from 512x512
    pred = zoom(pred, [curr_shape[0] / img_size, curr_shape[1] / img_size, 1.0], order=1)
    pred = (pred >= 0.5).astype(np.float32)

    print("morphological post-processing...")
    # morpological post-processing
    # 1) first erode
    pred = binary_erosion(pred.astype(bool), ball(3)).astype(np.float32)

    # 2) keep only largest connected component
    labels = label(pred)
    regions = regionprops(labels)
    area_sizes = []
    for region in regions:
        area_sizes.append([region.label, region.area])
    area_sizes = np.array(area_sizes)
    tmp = np.zeros_like(pred)
    tmp[labels == area_sizes[np.argmax(area_sizes[:, 1]), 0]] = 1
    pred = tmp.copy()
    del tmp, labels, regions, area_sizes

    # 3) dilate
    pred = binary_dilation(pred.astype(bool), ball(3))

    # 4) remove small holes
    pred = remove_small_holes(pred.astype(bool), area_threshold=0.001*np.prod(pred.shape)).astype(np.float32)

    print("saving...")
    pred = pred.astype(np.uint8)
    img = nib.Nifti1Image(pred, affine=resampled_volume.affine)
    resampled_lab = resample_from_to(img, nib_volume, order=0)
    nib.save(resampled_lab, output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', metavar='--i', type=str, nargs='?',
                    help="set path of which image to use.")
    parser.add_argument('--output', metavar='--o', type=str, nargs='?',
                    help="set path to store the output.")
    parser.add_argument('--cpu', action='store_true',
                    help="force using the CPU even if a GPU is available.")
    ret = parser.parse_args(sys.argv[1:]); print(ret)

    if ret.cpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    if not tf.test.is_gpu_available():
        tf.config.set_visible_devices([], 'GPU')
        visible_devices = tf.config.get_visible_devices()
    else:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, enable=True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    if not ret.input.endswith(".nii"):
        raise ValueError("Image provided is not in the supported '.nii' format.")
    if not ret.output.endswith(".nii"):
        raise ValueError("Output name set is not in the supported '.nii' format.")

    # fix paths
    ret.input = ret.input.replace("\\", "/")
    ret.output = ret.output.replace("\\", "/")

    func(*vars(ret).values())


if __name__ == "__main__":
    main()
