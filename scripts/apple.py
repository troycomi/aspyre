import argparse
import os
import sys

from aspyre import config
from aspyre.apple.apple import Apple
from aspyre.apple.exceptions import ConfigError


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Apple Picker')
    parser.add_argument("-s", type=int, help="size of particle")
    parser.add_argument("--jpg", action='store_true', help="create result image")
    parser.add_argument("-o", type=str, metavar="output dir",
                        help="name of output folder where star file should be saved (by default "
                             "AP saves to input folder and adds 'picked' to original file name.)")

    parser.add_argument("mrcdir", type=str,
                        help="path to folder containing all mrc files to pick.")

    args = parser.parse_args()

    # A dictionary to override application configuration parameters
    override_dict = {}

    if args.s:
        override_dict['apple.particle_size'] = args.s

    if args.o:
        if not os.path.exists(args.o):
            raise ConfigError("Output directory doesn't exist! {}".format(args.o))
        override_dict['apple.output_dir'] = args.o

    if args.jpg:
        override_dict['apple.create_jpg'] = True

    if not os.path.exists(args.mrcdir):
        print("mrc folder {} doesn't' exist! terminating..".format(args.mrcdir))
        sys.exit(1)

    if not os.listdir(args.mrcdir):
        print("mrc folder is empty! terminating..")
        sys.exit(1)

    with config.override(override_dict):
        apple = Apple(args.mrcdir)
        apple.pick_particles()
