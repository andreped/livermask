import gdown
import logging as log
import chainer
from .unet3d import UNet3D


def get_model(output):
    url = "https://drive.google.com/uc?id=12or5Q79at2BtLgQ7IaglNGPFGRlEgEHc"
    md5 = "ef5a6dfb794b39bea03f5496a9a49d4d"
    gdown.cached_download(url, output) #, md5=md5) #, postprocess=gdown.extractall)


def get_vessel_model(output):
    url = "https://drive.google.com/uc?id=1-8VNoRmIeiF1uIuWBqmZXz_6dIQFSAxN"
    #md5 = "ef5a6dfb794b39bea03f5496a9a49d4d"
    gdown.cached_download(url, output) #, md5=md5)


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
