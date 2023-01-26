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
import logging as log
import chainer
import math
from .unet3d import UNet3D
from .yaml_utils import Config
import yaml
from tensorflow.keras import backend as K
from numba import cuda
from .utils import load_vessel_model
import multiprocessing as mp


def intensity_normalization(volume, intensity_clipping_range):
    result = np.copy(volume)

    result[volume < intensity_clipping_range[0]] = intensity_clipping_range[0]
    result[volume > intensity_clipping_range[1]] = intensity_clipping_range[1]

    min_val = np.amin(result)
    max_val = np.amax(result)
    if (max_val - min_val) != 0:
        result = (result - min_val) / (max_val - min_val)
    return result


def liver_segmenter_wrapper(curr, output, cpu, verbose, multiple_flag, name, extension):
    # run inference in a different process
    mp.set_start_method('spawn', force=True)
    with mp.Pool(processes=1, maxtasksperchild=1) as p:  # , initializer=initializer)
        result = p.map_async(liver_segmenter, ((curr, output, cpu, verbose, multiple_flag, name, extension),))
        log.info("getting result from process...")
        ret = result.get()[0]
    return ret


def liver_segmenter(params):
    try:
        curr, output, cpu, verbose, multiple_flag, name, extension = params

        # load model
        model = load_model(name, compile=False)

        log.info("preprocessing...")
        nib_volume = nib.load(curr)
        new_spacing = [1., 1., 1.]
        resampled_volume = resample_to_output(nib_volume, new_spacing, order=1)
        data = resampled_volume.get_data().astype('float32')

        curr_shape = data.shape

        # resize to get (512, 512) output images
        img_size = 512
        data = zoom(data, [img_size / data.shape[0], img_size / data.shape[1], 1.0], order=1)

        # intensity normalization
        intensity_clipping_range = [-150, 250]  # HU clipping limits (Pravdaray's configs)
        data = intensity_normalization(volume=data, intensity_clipping_range=intensity_clipping_range)

        # fix orientation
        data = np.rot90(data, k=1, axes=(0, 1))
        data = np.flip(data, axis=0)

        log.info("predicting...")
        # predict on data
        pred = np.zeros_like(data).astype(np.float32)
        for i in tqdm(range(data.shape[-1]), "pred: ", disable=not verbose):
            pred[..., i] = \
            model.predict(np.expand_dims(np.expand_dims(np.expand_dims(data[..., i], axis=0), axis=-1), axis=0))[
                0, ..., 1]
        del data

        # threshold
        pred = (pred >= 0.4).astype(int)

        # fix orientation back
        pred = np.flip(pred, axis=0)
        pred = np.rot90(pred, k=-1, axes=(0, 1))

        log.info("resize back...")
        # resize back from 512x512
        pred = zoom(pred, [curr_shape[0] / img_size, curr_shape[1] / img_size, 1.0], order=1)
        pred = (pred >= 0.5).astype(bool)

        log.info("morphological post-processing...")
        # morpological post-processing
        # 1) first erode
        pred = binary_erosion(pred, ball(3)).astype(np.int32)

        # 2) keep only largest connected component
        labels = label(pred)
        nb_uniques = len(np.unique(labels))  # note: includes background 0
        if nb_uniques > 2:  # if only one, no filtering needed
            regions = regionprops(labels)
            area_sizes = []
            for region in regions:
                area_sizes.append([region.label, region.area])
            area_sizes = np.array(area_sizes)
            tmp = np.zeros_like(pred)
            tmp[labels == area_sizes[np.argmax(area_sizes[:, 1]), 0]] = 1
            pred = tmp.copy()
            del tmp, labels, regions, area_sizes
        
        if nb_uniques > 1:  # if no segmentation, no post-processing needed
            # 3) dilate
            pred = binary_dilation(pred.astype(bool), ball(3))

            # 4) remove small holes
            pred = remove_small_holes(pred.astype(bool), area_threshold=0.001 * np.prod(pred.shape))

        log.info("saving...")
        pred = pred.astype(np.uint8)
        img = nib.Nifti1Image(pred, affine=resampled_volume.affine)
        resampled_lab = resample_from_to(img, nib_volume, order=0)
        if multiple_flag:
            nib.save(resampled_lab, output + "/" + curr.split("/")[-1].split(".")[0] + "-livermask" + extension)
        else:
            nib.save(resampled_lab, output + "-livermask" + extension)

        return pred
    except KeyboardInterrupt:
        raise "Caught KeyboardInterrupt, terminating worker"


def vessel_segmenter(curr, output, cpu, verbose, multiple_flag, liver_mask, name_vessel, extension):
    # check if cupy is available, if not, set cpu=True
    try:
        import cupy
    except ModuleNotFoundError as e:
        log.info(e)
        log.info("cupy is not available. Setting cpu=True")
        cpu = True

    # load model
    unet, xp = load_vessel_model(name_vessel, cpu)

    # read config
    config = Config(yaml.safe_load(open(os.path.dirname(os.path.abspath(__file__)) + "/../configs/base.yml")))

    log.info("resize back...")
    nib_volume = nib.load(curr)
    new_spacing = [1., 1., 1.]
    resampled_volume = resample_to_output(nib_volume, new_spacing, order=1)
    # resampled_volume = nib_volume
    org = resampled_volume.get_data().astype('float32')

    # HU clipping
    intensity_clipping_range = [80, 220]
    org[org < intensity_clipping_range[0]] = intensity_clipping_range[0]
    org[org > intensity_clipping_range[1]] = intensity_clipping_range[1]

    # Calculate maximum of number of patch at each side
    ze, ye, xe = org.shape
    xm = int(math.ceil((float(xe) / float(config.patch['patchside']))))
    ym = int(math.ceil((float(ye) / float(config.patch['patchside']))))
    zm = int(math.ceil((float(ze) / float(config.patch['patchside']))))

    margin = ((0, config.patch['patchside']),
              (0, config.patch['patchside']),
              (0, config.patch['patchside']))
    org = np.pad(org, margin, 'edge')
    org = chainer.Variable(xp.array(org[np.newaxis, np.newaxis, :], dtype=xp.float32))

    # init prediction array
    prediction_map = np.zeros(
        (ze + config.patch['patchside'], ye + config.patch['patchside'], xe + config.patch['patchside']))
    probability_map = np.zeros((config.unet['number_of_label'], ze + config.patch['patchside'],
                                ye + config.patch['patchside'], xe + config.patch['patchside']))

    log.info("predicting...")
    # Patch loop
    for s in tqdm(range(xm * ym * zm), 'Patch loop', disable=not verbose):
        xi = int(s % xm) * config.patch['patchside']
        yi = int((s % (ym * xm)) / xm) * config.patch['patchside']
        zi = int(s / (ym * xm)) * config.patch['patchside']

        # check if current region contains any liver mask, if not, skip
        parenchyma_patch = liver_mask[zi:zi + config.patch['patchside'], yi:yi + config.patch['patchside'],
                                      xi:xi + config.patch['patchside']]
        # if np.count_nonzero(parenchyma_patch) == 0:
        if np.mean(parenchyma_patch) < 0.25:
            continue

        # Extract patch from original image
        patch = org[:, :, zi:zi + config.patch['patchside'], yi:yi + config.patch['patchside'],
                    xi:xi + config.patch['patchside']]
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            probability_patch = unet(patch)

        # Generate probability map
        probability_patch = probability_patch.data
        # if args.gpu >= 0:
        if not cpu:
            probability_patch = chainer.cuda.to_cpu(probability_patch)
        for ch in range(probability_patch.shape[1]):
            probability_map[ch, zi:zi + config.patch['patchside'], yi:yi + config.patch['patchside'],
                            xi:xi + config.patch['patchside']] = probability_patch[0, ch, :, :, :]

        prediction_patch = np.argmax(probability_patch, axis=1)

        prediction_map[zi:zi + config.patch['patchside'], yi:yi + config.patch['patchside'],
                       xi:xi + config.patch['patchside']] = prediction_patch[0, :, :, :]

    # probability_map = probability_map[:, :ze, :ye, :xe]
    prediction_map = prediction_map[:ze, :ye, :xe]

    # post-process prediction
    # prediction_map = prediction_map + liver_mask
    # prediction_map[prediction_map > 0] = 1

    # filter segmented vessels outside the predicted liver parenchyma
    pred = prediction_map.astype(np.uint8)
    pred[liver_mask == 0] = 0

    log.info("saving...")
    img = nib.Nifti1Image(pred, affine=resampled_volume.affine)
    resampled_lab = resample_from_to(img, nib_volume, order=0)
    if multiple_flag:
        nib.save(resampled_lab, output + "/" + curr.split("/")[-1].split(".")[0] + "-vessels" + extension)
    else:
        nib.save(resampled_lab, output + "-vessels" + extension)
