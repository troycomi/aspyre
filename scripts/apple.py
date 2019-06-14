from aspyre.apple.apple import Apple
from aspyre.utils.config import ConfigArgumentParser


if __name__ == '__main__':

    parser = ConfigArgumentParser(description='Apple Picker')
    parser.add_argument("mrc_dir", help="Path to folder containing all mrc files to pick.")
    parser.add_argument("--output_dir",
                        help="Path to folder to save *.star files. If unspecified, no star files are created.")
    parser.add_argument("--create_jpg", action='store_true', help="save *.jpg files for picked particles.")

    with parser.parse_args() as args:
        apple = Apple(args.mrc_dir, args.output_dir, args.create_jpg)
        apple.pick_particles()
