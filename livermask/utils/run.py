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
import math
from .unet3d import UNet3D
import yaml
from tensorflow.keras import backend as K
from numba import cuda
from .process import liver_segmenter_wrapper, vessel_segmenter, intensity_normalization
from .utils import verboseHandler
import logging as log
from .utils import get_model, get_vessel_model


def run_analysis(path, output, cpu, verbose, vessels, extension, name=None, name_vessel=None, mp_enabled=True):
    # fix paths (necessary if called as a package and not CLI)
    path = path.replace("\\", "/")
    output = output.replace("\\", "/")

    if cpu:
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

    # enable verbose or not
    log = verboseHandler(verbose)
    
    # if model names are not provided, download them (necessary for docker,
    # where we cannot perform HTTP requests from inside container)
    cwd = "/".join(os.path.realpath(__file__).replace("\\", "/").split("/")[:-1]) + "/"
    log.info("Model names: " + str(name) + ", " + str(name_vessel))
    if name is None:
        name = cwd + "model.h5"
        get_model(name)
    
    if vessels and name_vessel is None:
        name_vessel = cwd + "model-hepatic_vessel.npz"
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
            pred = liver_segmenter_wrapper(curr, output, cpu, verbose, multiple_flag, name, extension, mp_enabled)

            if vessels:
                # perform liver vessel segmentation
                vessel_segmenter(curr, output, cpu, verbose, multiple_flag, pred, name_vessel, extension)
        else:
            log.info("Unsupported file: " + curr)
