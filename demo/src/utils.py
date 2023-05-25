import nibabel as nib
import numpy as np


def load_ct_to_numpy(data_path):
    if type(data_path) != str:
        data_path = data_path.name

    image = nib.load(data_path)
    data = image.get_fdata()

    data = np.rot90(data, k=1, axes=(0, 1))

    data[data < -150] = -150
    data[data > 250] = 250

    data = data - np.amin(data)
    data = data / np.amax(data) * 255
    data = data.astype("uint8")

    print(data.shape)
    return [data[..., i] for i in range(data.shape[-1])]


def load_pred_volume_to_numpy(data_path):
    if type(data_path) != str:
        data_path = data_path.name

    image = nib.load(data_path)
    data = image.get_fdata()

    data = np.rot90(data, k=1, axes=(0, 1))

    data[data > 0] = 1
    data = data.astype("uint8")

    print(data.shape)
    return [data[..., i] for i in range(data.shape[-1])]
