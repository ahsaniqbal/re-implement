"""Tests ImageNetUtils module.

This module tests ImageNetUtils, from datasets/image_net/ImageNetUtils.
"""

import os
import os.path as osp
import pytest
import pandas as pd
from datasets.classification.image_net import utils

def test_create_csv_for_invalid_input() -> None:
    """Tests the function `create_csv` for invalid input.

    Args:
        There are no arguments.

    Returns:
        The function doesn't return anything. 
    """

    current_working_dir = osp.abspath('./')
    temp_test_path = osp.join(current_working_dir, 'temp_test_data')

    if not osp.exists(temp_test_path):
        os.makedirs(temp_test_path)

    with open(osp.join(temp_test_path, 'temp.txt'), 'w') as file_ptr:
        file_ptr.write('This is temp file.')

    with pytest.raises(IOError):
        utils.create_csv('/home/nouser/this/path/does/not/exist')

    with pytest.raises(IOError):
        utils.create_csv(osp.join(temp_test_path, 'temp.txt'))

def test_create_csv_for_valid_input() -> None:
    """Tests the function `create_csv` for valid input.
    
    Args:
        There are no arguments.

    Returns:
        The function doesn't return anything.
    """

    current_working_dir = osp.abspath('./')
    temp_test_path = osp.join(current_working_dir, 'temp_test_data')

    if not os.path.exists(osp.join(temp_test_path, '00000')):
        os.makedirs(osp.join(temp_test_path, '00000'))

    with open(osp.join(temp_test_path, '00000', 'temp.jpg'), 'wb') as file_ptr:
        file_ptr.write('This is temp file.'.encode())

    utils.create_csv(temp_test_path)
    assert osp.exists(osp.join(temp_test_path, 'image-net.csv')) and\
        osp.isfile(osp.join(temp_test_path, 'image-net.csv'))

    data = pd.read_csv(osp.join(osp.join(temp_test_path, 'image-net.csv')), header=None)
    assert len(data) > 0

    assert data.iloc[0, 0].strip() == osp.join(temp_test_path, '00000', 'temp.jpg')
    assert data.iloc[0, 1] == 0


def test_merge_data_for_invalid_input() -> None:
    """Tests the function `merge_data` for invalid input.

    Args:
        There are no arguments.

    Returns:
        The function doesn't return anything.
    """

    current_working_dir = osp.abspath('./')
    temp_test_path = osp.join(current_working_dir, 'temp_test_data')

    if not os.path.exists(temp_test_path):
        os.makedirs(temp_test_path)

    with open(osp.join(temp_test_path, 'temp.txt'), 'w') as file_ptr:
        file_ptr.write('This is temp file')

    with pytest.raises(IOError): # both path don't exist.
        utils.merge_data('/home/nouser/this/path/does/not/exist',
                                 '/home/nouser/this/path/does/not/exist')

    with pytest.raises(IOError):
        # the first path exist but it is a file, second path doesn't exist.
        utils.merge_data(osp.join(temp_test_path, 'temp.txt'),
                                 '/home/nouser/this/path/does/not/exist')

    with pytest.raises(IOError): # first argument/path is valid, second path doesn't exist.
        utils.merge_data(temp_test_path, '/home/nouser/this/path/does/not/exist')

    with pytest.raises(IOError):
        # first argument/path is valid, second path exists but it is a file.
        utils.merge_data(temp_test_path, osp.join(temp_test_path, 'temp.txt'))
