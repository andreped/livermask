import numpy as np
import os
import sys
from tqdm import tqdm 
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to
from scipy.ndimage import zoom
from tensorflow.python.keras.models import load_model
from skimage.morphology import remove_small_holes, binary_dilation, binary_erosion, ball
from skimage.measure import label, regionprops
import warnings
import argparse
import pkg_resources
import tensorflow as tf
import logging as log
import chainer
import math
from .utils.unet3d import UNet3D
import yaml
from tensorflow.keras import backend as K
from numba import cuda
from .utils.process import liver_segmenter_wrapper, vessel_segmenter, intensity_normalization
from .utils.utils import verboseHandler
import logging as log
from .utils.utils import get_model, get_vessel_model


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # due to this: https://github.com/tensorflow/tensorflow/issues/35029
warnings.filterwarnings('ignore', '.*output shape of zoom.*')  # mute some warnings


def func(path, output, cpu, verbose, vessels, extension):
    # enable verbose or not
    log = verboseHandler(verbose)

    cwd = "/".join(os.path.realpath(__file__).replace("\\", "/").split("/")[:-1]) + "/"
    name = cwd + "model.h5"
    name_vessel = cwd + "model-hepatic_vessel.npz"

    # get models
    get_model(name)

    if vessels:
        get_vessel_model(name_vessel)

    if not os.path.isdir(path):
        paths = [path]
    else:
        paths = [path + "/" + p for p in os.listdir(path)]

    multiple_flag = len(paths) > 1
    if multiple_flag:
        os.makedirs(output + "/", exist_ok=True)

    for curr in tqdm(paths, "CT:"):
        # check if current file is a nifti file, if not, skip
        if curr.endswith(".nii") or curr.endswith(".nii.gz"):
            # perform liver parenchyma segmentation, launch it in separate process to properly clear memory
            pred = liver_segmenter_wrapper(curr, output, cpu, verbose, multiple_flag, name, extension)

            if vessels:
                # perform liver vessel segmentation
                vessel_segmenter(curr, output, cpu, verbose, multiple_flag, pred, name_vessel, extension)
        else:
            log.info("Unsuported file: " + curr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', metavar='--i', type=str, nargs='?',
                    help="set path of which image(s) to use.")
    parser.add_argument('--output', metavar='--o', type=str, nargs='?',
                    help="set path to store the output.")
    parser.add_argument('--cpu', action='store_true',
                    help="force using the CPU even if a GPU is available.")
    parser.add_argument('--verbose', action='store_true',
                    help="enable verbose.")
    parser.add_argument('--vessels', action='store_true',
                    help="segment vessels.")
    parser.add_argument('--extension', metavar='--e', type=str, default=".nii",
                    help="define the output extension. (default: .nii)")
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

    if ret.input is None:
        raise ValueError("Please, provide an input.")
    if ret.output is None:
        raise ValueError("Please, provide an output.")

    # fix paths
    ret.input = ret.input.replace("\\", "/")
    ret.output = ret.output.replace("\\", "/")

    if not os.path.isdir(ret.input) and not ret.input.endswith(".nii") and not ret.input.endswith(".nii.gz"):
        raise ValueError("Input path provided is not in the supported '.nii' or '.nii.gz' formats or a directory.")
    if ret.output.endswith(".nii") or ret.output.endswith(".nii.gz") or "." in ret.output.split("/")[-1]:
        raise ValueError("Output path provided is not a directory or a name (remove *.nii format from name).")
    if ret.extension not in [".nii", ".nii.gz"]:
        raise ValueError("Extension not supported. Expected: .nii or .nii.gz")

    func(*vars(ret).values())


if __name__ == "__main__":
    main()
