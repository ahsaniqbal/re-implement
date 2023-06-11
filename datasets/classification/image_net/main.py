"""The module is used as a driver/main to run image-net utilities.
"""
import argparse
from datasets.classification.image_net import utils


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=int,
                        help="For creating csv, the mode should be 1, \
                            for merging data mode should be 2.")
    parser.add_argument("base_path", type=str,
                        help="For mode=1, this argument is used for reading \
                            the list of files in the dataset, and also the \
                            resultant csv is written at this path. For mode=2, \
                            this is the source path of files to be moved in destination.")
    parser.add_argument("dest_path", type=str,
                        help="Used only when mode=2, the path where files will be moved.")
    parser.add_argument("num_rand_classes", type=int,
                        help="Only used for mode=3, this param is used for generating the\
                            subset of the dataset.")
    parser.add_argument("num_samples_per_classes", type=int,
                        help="Only used for mode=3, this param is used for generating the\
                            subset of the dataset.")
    args = parser.parse_args()

    if args.mode == 1:
        utils.create_csv(base_path=args.base_path)
    elif args.mode == 2:
        utils.merge_data(dest_base_path=args.dest_path, source_base_path=args.base_path)
    elif args.mode == 3:
        utils.generate_subset_csv(source_path=args.base_path,
                                  dest_path=args.dest_path,
                                  num_classes=args.num_rand_classes,
                                  num_samples=args.num_samples_per_classes)
    else:
        raise ValueError(f"Unrecognised mode:{args.mode}")
    