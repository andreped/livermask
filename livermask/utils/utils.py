import gdown
import logging as log
import chainer
from .unet3d import UNet3D
from .fetch import download


def get_model(output):
    url = "https://github.com/andreped/livermask/releases/download/trained-models-v1/model.h5"
    download(url, output)


def get_vessel_model(output):
    url = "https://drive.google.com/uc?id=1-8VNoRmIeiF1uIuWBqmZXz_6dIQFSAxN"
    gdown.cached_download(url, output)


def load_vessel_model(path, cpu):
    unet = UNet3D(num_of_label=2)
    chainer.serializers.load_npz(path, unet)
    if not cpu:
        chainer.cuda.get_device_from_id(0).use()
        unet.to_gpu()
    xp = unet.xp
    return unet, xp


def verboseHandler(verbose):
    if verbose:
        log.basicConfig(format="%(levelname)s: %(message)s", level=log.DEBUG)
        log.info("Verbose output.")
    else:
        log.basicConfig(format="%(levelname)s: %(message)s")
    return log
