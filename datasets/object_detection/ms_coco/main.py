"""The module is used as a driver/main to run ms-coco utilities.
"""
import argparse
from datasets.object_detection.ms_coco import utils

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", type=int,
                        help="For creating pickle file, the mode should be 1, \
                            for visualization of random sample mode should be 2.")
    parser.add_argument("base_path", type=str,
                        help="The root path of the dataset.")
    parser.add_argument("split", type=str,
                        help="The dataset split should be in (train, test, val).")
    args = parser.parse_args()

    if args.mode == 1:
        utils.create_pickle(args.base_path, args.split)
    elif args.mode == 2:
        utils.visualize_random_sample(args.base_path, args.split)
    else:
        raise ValueError(f"Invalid mode `{args.mode}`.")
