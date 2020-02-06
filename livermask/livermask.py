import numpy as np 
import os, sys
import h5py
from tqdm import tqdm 
import nibabel as nib
from nibabel.processing import resample_to_output, resample_from_to
from scipy.ndimage import zoom
from tensorflow.python.keras.models import load_model
import gdown
import warnings
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

def get_model():
	url = "https://drive.google.com/uc?id=1RBhmmpeH-pd6ugV5-y_9coEoVHJlw_b-"
	output = "model.h5"
	md5 = "aebe1d94c52abfd7468da0f189a49c47"
	gdown.cached_download(url, output, md5=md5, postprocess=gdown.extractall)

def func(path, output):

	cwd = "/".join(os.path.realpath(__file__).replace("\\", "/").split("/")[:-1]) + "/"

	#print(cwd)
	#print(" :) ")

	name = cwd + "model.h5"
	#name = "\.model.h5"

	# get model
	get_model()

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

	# threshold
	pred = (pred >= 0.5).astype(int)

	# fix orientation back
	pred = np.flip(pred, axis=0)
	pred = np.rot90(pred, k=-1, axes=(0, 1))

	print("postprocessing...")
	# resize back from 512x512
	pred = zoom(pred, [curr_shape[0] / img_size, curr_shape[1] / img_size, 1.0], order=1)
	pred = (pred >= 0.5).astype(np.float32)

	print("saving...")
	pred = pred.astype(np.uint8)
	img = nib.Nifti1Image(pred, affine=resampled_volume.affine)
	resampled_lab = resample_from_to(img, nib_volume, order=0)
	nib.save(resampled_lab, output)


def main():
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	#__os.path

	path = sys.argv[1]
	output = sys.argv[2]

	func(path, output)


if __name__ == "__main__":
	main()


	


