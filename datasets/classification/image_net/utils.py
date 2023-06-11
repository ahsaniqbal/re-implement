"""Utility to transform image net dataset into a csv for pytorch dataset.
    Also merging data from two directories into a single directory.

This utility is used to transform image net dataset contained in a directory 
to a csv file. The csv file will contain two columns, first column will contain the path 
of the image on disk and second column will have the target label of the image.
Also this utility could be used to merge dataset from two folders into one. 
The utility could be invoked from command line, as well the functions could be invoked directly.

usage example:
  create_csv(path/to/data)
  merge_data(path/to/dest, path/to/source)
"""

import os
from os import path as osp
import shutil
import random
import pandas as pd


def create_csv(base_path: str) -> None:
    """Creates a csv representing the dataset.

    Creates and writes csv representing the dataset on the disk.
    The dataset is supposed to be in a directory structure like base_path/class/images.jpg.
    The resultant csv has two columns, the first column contains path to the image, and the
    second column contains the target column of the image.

    Args:
        base_path: String representing the base_path to the dataset on the disk.

    Returns:
        The function returns nothing.

    Raises:
        IOError: An error is raised if base_path doesn't exist or it doesn't represent a directory.
    """

    if not osp.exists(base_path) or not osp.isdir(base_path):
        raise IOError(f"The path '{base_path}' does not exist or is not a directory.")

    sub_dirs = [d for d in os.listdir(base_path) if osp.isdir(osp.join(base_path, d))]

    data = []
    for dir_name in sub_dirs:
        path = osp.join(base_path, dir_name)
        data += [
            (osp.join(path, img), int(dir_name.strip()))
            for img in os.listdir(path)
            if osp.isfile(osp.join(path, img)) and img.endswith(".jpg")
        ]

    data_frame = pd.DataFrame(data, columns=["image", "label"])
    data_frame.to_csv(osp.join(base_path, "image-net.csv"), index=False, header=False)


def merge_data(dest_base_path: str, source_base_path: str) -> None:
    """Merges the data from two locations into one.

    Merges the data from two locations on the disk into one. Both directories should have same
    structure like base_path/class/images.jpg.

    Args:
        dest_base_path: str, reprsenting the destination path on the disk,
            where the merged dataset will be.
        source_base_path: str, representing the source path on the disk,
            from where data has to be moved.

    Returns:
        The funstion returns nothing.

    Raises:
        IOError: An error is raised if any of arguments doesn't exist on disk,
            or doesn't represent the directory.
    """

    if not osp.exists(dest_base_path) or not osp.isdir(dest_base_path):
        raise IOError(
            f"The path '{dest_base_path}' does not exist or is not a directory."
        )
    if not osp.exists(source_base_path) or not osp.isdir(source_base_path):
        raise IOError(
            f"The path '{source_base_path}' does not exist or is not a directory."
        )

    sub_dirs = [
        d
        for d in os.listdir(source_base_path)
        if osp.isdir(osp.join(source_base_path, d))
    ]

    for dir_name in sub_dirs:
        path = osp.join(source_base_path, dir_name)
        images = [
            img
            for img in os.listdir(path)
            if osp.isfile(osp.join(path, img)) and img.endswith(".jpg")
        ]

        for img in images:
            source_path = osp.join(source_base_path, dir, img)
            dest_path = osp.join(dest_base_path, dir, img)

            if not osp.exists(dest_path):
                shutil.move(source_path, dest_path)


def generate_subset_csv(
    source_path: str, dest_path: str, num_classes: int, num_samples: int
) -> None:
    """Generates the subset of the dataset.

    Args:
        source_path: the path to the csv, reprsenting the full dataset.
        dest_path: the path to the destination csv.

    Returns:
        The function returns nothing.

    Raises:
        An error is raised if any of base_path doesn't exist on disk, or doesn't represent the file.
    """

    if not osp.exists(source_path) or not osp.isdir(source_path):
        raise IOError(f"The path '{source_path}' doesn't exist or is not a directory.")
    if not osp.exists(dest_path) or not osp.isdir(dest_path):
        raise IOError(f"The path '{dest_path}' doesn't exist or is not a directory.")

    data = pd.read_csv(osp.join(source_path, "image-net.csv"))

    files = []
    targets = []

    classes_used = set([])
    for idx in range(num_classes):
        class_index = random.randint(0, 999)
        while class_index in classes_used:
            class_index = random.randint(0, 999)
        classes_used.add(class_index)

        data_frame = pd.DataFrame(data.loc[data.iloc[:, 1] == class_index])

        files += [data_frame.iloc[i, 0] for i in range(min(len(data_frame), num_samples))]
        targets += [idx] * min(len(data_frame), num_samples)

    data = {"file": files, "target": targets}
    data_frame = pd.DataFrame(data)
    data_frame.to_csv(
        osp.join(dest_path, f"image-net-{num_classes}.csv"), index=False, header=False
    )
