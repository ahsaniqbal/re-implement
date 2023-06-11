"""Module implements image net dataset.

Typical usage example:
dataset = ImageNet(csv_file="/path/to/csv/csv-file.csv",
                       root_dir="/path/to/root/of/dataset",
                       transform=None)
"""
from typing import Tuple, Optional

import torch
import pandas as pd

from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import Compose


class ImageNet(Dataset):
    """ImageNet dataset.

    Attributes:
        csv_data: pandas.DataFrame, represents csv data, csv has two column,
            the image path and its target label.
        transform: torchvision.transform.Compose, preprocessing transforms. 
    """

    def __init__(self, csv_file: str, transform: Optional[Compose]=None) -> None:
        """Initializes the dataset.
        
        Args:
            csv_file: str, path to the csv file representing the dataset.
            transform: torchvision.transform.Compose, preprocessing transforms.

        Returns:
            The method returns nothing.
        """
        super().__init__()
        self.csv_data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self) -> int:
        """Returns the length of the dataset.
        
        Args:
            The method doesn't expect any argument.

        Returns:
            An integer representing the length of the dataset.
        """
        return len(self.csv_data)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Returns the sample from the dataset given its index.

        Args:
            idx: int, the index of the sample.

        Returns:
            A tuple T(torch.Tensor, int).
                T[0] is a tensor of size (C, H, W), where C, H, W are channels, height, and width.
                T[1] is an integer, which represents the target label of the sample.
        """
        image_name = self.csv_data.iloc[idx, 0].strip()
        image_label = self.csv_data.iloc[idx, 1]

        image = Image.open(image_name)  # io.imread(image_name)

        if self.transform:
            image = self.transform(image)
        return (image, image_label)
