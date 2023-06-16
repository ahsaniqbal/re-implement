"""Utility to transform image net dataset into a pickle file for pytorch dataset.
    Also used to visualize bounding boxes on top of images.

usage example:
  create_pickle(path/to/data, split)
  visualize_random_sample(path/to/data, split)
"""
import os
import pickle
import random
from numbers import Number
from typing import Dict, List, Mapping
from PIL import Image, ImageDraw

def read_targets(path_to_label: str) -> List[Dict[str, Number]]:
    """Reads the tagets given the path to the label file.
    
    Args:
        path_to_label: str, a string representing the file containing target information.

    Raises:
        A ValueError if the label file doesn't exist.

    Returns:
        A dict object with keys as strings and values as numbers.
    """
    if not os.path.exists(path_to_label):
        raise ValueError(f"The label file `{path_to_label}` doesn't exist.")

    result: List[Mapping[str, Number]] = []
    with open(path_to_label, "r") as data_file:
        targets = data_file.readlines()
        targets = [target.strip() for target in targets if len(target.strip()) > 0]
        for target in targets:
            target = target.strip().split()
            if len(target) != 5:
                raise ValueError(f"Invalid target, it should have 5 numbers, found{len(target)}.")
            result.append({
                'label-id': int(target[0].strip()),
                'bbox-center-x': float(target[1].strip()),
                'bbox-center-y': float(target[2].strip()),
                'bbox-width': float(target[3].strip()),
                'bbox-height': float(target[4].strip()),
            })
    return result

def create_pickle(base_path: str, split: str) -> None:
    """Creates a dataset json.

    Represents dataset as one json file. The json looks like 
    data : 
    [
        {
            'image': /path/to/image,
            'targets':
            [
                {
                    'label-id': 0,
                    'bbox-center-x': 0.25,
                    'bbox-center-y': 0.25,
                    'bbox-width': 0.4,
                    'bbox-height': 0.6,
                },
            ]
        },
    ]

    Args:
        base_path: str, a string representing base path of the dataset.
        split: str, a string representing the dataset split.
            The valid value is among [train|test|val].

    Raises:
        A ValueError is raised if split is not valid.
        A ValueError is raised if the base_path doesn't exist.

    Returns:
        The function doesn't return anything.
    """
    if not os.path.exists(base_path) or not os.path.isdir(base_path):
        raise ValueError(f"The path `{base_path}` doesn't exist or it is not a directory.")

    if split not in ["train", "test", "val"]:
        raise ValueError(f"The split `{split}` is not valid, it must be in [train, test, val].")

    path_to_images = os.path.join(base_path, "images", f"{split}2014")
    if not os.path.exists(path_to_images):
        raise ValueError(f"The path to images: `{path_to_images}` doesn't exist.")

    path_to_labels = os.path.join(base_path, "labels", f"{split}2014")
    if split != "test" and not os.path.exists(path_to_labels):
        raise ValueError(f"The path to labels: `{path_to_labels}` doesn't exist.")

    result = []
    no_labels = 0
    for img_name in os.listdir(path_to_images):
        sample = {}

        sample["image"] = img_name.strip()
        if split != "test":
            label_file = os.path.join(path_to_labels, f"{img_name.strip().split('.')[0]}.txt")
            try:
                sample["targets"] = read_targets(label_file)
            except ValueError:
                no_labels += 1
                continue
        result.append(sample)
    print(f"No Label Found:{no_labels}")
    with open(os.path.join(base_path, f"ms-coco-{split}.pkl"), "wb") as data_file:
        pickle.dump(result, data_file)

def visualize_random_sample(base_path: str, split: str) -> None:
    """Visualizes a random sample with object bounding boxes.

    Args:
        base_path: str, the path to the dataset, should contain the pickle file.
        split: str, the dataset split, should be one of (train, test, val).

    Raises:
        A ValueError, if ms-coco-{split}.pkl not found at base_path,
            or images/{split}2014 not found at base_path.
        A ValueError, if split is not one of (train, test, val).


    Returns:
        Function doesn't return anything.
    """

    if split not in ["train", "test", "val"]:
        raise ValueError(f"Invalid split, it should be one of (train, test, val), found `{split}`.")

    if not os.path.exists(base_path)\
        or not os.path.exists(os.path.join(base_path, f"ms-coco-{split}.pkl"))\
        or not os.path.exists(os.path.join(base_path, "images", f"{split}2014")):
        raise ValueError(f"The path `{base_path}` doesn't exist," +
                         f" or pickle file not found under `{base_path}`,"+
                         f" or images/{split}2014 not found under `{base_path}`.")

    with open(os.path.join(base_path, f"ms-coco-{split}.pkl"), "rb") as data_file:
        data = pickle.load(data_file)

    random_index = random.randint(0, len(data) - 1)
    random_sample = data[random_index]

    image = Image.open(os.path.join(base_path, "images", f"{split}2014", random_sample["image"]))
    image_width, image_height = image.size
    canvas = ImageDraw.Draw(image)


    for target in random_sample["targets"]:
        bbox_center_x = int(image_width * target["bbox-center-x"])
        bbox_center_y = int(image_height * target["bbox-center-y"])

        bbox_width = int(image_width * target["bbox-width"])
        bbox_height = int(image_height * target["bbox-height"])

        top, left = bbox_center_y - bbox_height // 2, bbox_center_x - bbox_width // 2
        bot, right= bbox_center_y + bbox_height // 2, bbox_center_x + bbox_width // 2

        canvas.rectangle((left, top, right, bot),
                         fill= None, 
                         outline=(random.randint(0, 255),
                                  random.randint(0, 255),
                                  random.randint(0, 255)),
                         width=2)
    image.show()
