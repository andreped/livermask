import sys
import os
import warnings
import argparse


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # due to this: https://github.com/tensorflow/tensorflow/issues/35029
warnings.filterwarnings('ignore', '.*output shape of zoom.*')  # mute some warnings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, nargs='?',
                    help="set path of which image(s) to use.")
    parser.add_argument('-o', '--output', type=str, nargs='?',
                    help="set path to store the output.")
    parser.add_argument('-c', '--cpu', action='store_true',
                    help="force using the CPU even if a GPU is available.")
    parser.add_argument('-v', '--verbose', action='store_true',
                    help="enable verbose.")
    parser.add_argument('-vs', '--vessels', action='store_true',
                    help="segment vessels.")
    parser.add_argument('-e', '--extension', type=str, default=".nii",
                    help="define the output extension. (default: .nii)")
    ret = parser.parse_args(sys.argv[1:])
    print(ret)

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

    # finally, import run_analysis method with relevant imports and run analysis
    from .utils.run import run_analysis
    run_analysis(*vars(ret).values())


if __name__ == "__main__":
    main()
