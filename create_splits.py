import argparse
import glob
import os
import random

import numpy as np

from utils import get_module_logger

TRAIN_RATIO=0.8
EVAL_RATIO=0.1
TEST_RATIO=0.2

def split(source, destination):
    """
    Create three splits from the processed records. The files should be moved to new folders in the
    same directory. This folder should be named train, val and test.

    args:
        - source [str]: source data directory, contains the processed tf records
        - destination [str]: destination data directory, contains 3 sub folders: train / val / test
    """
    files = glob.glob(source)
    shuffle(files)
    number_files = len(files)
    
    os.makedirs(destination + "train", exist_ok=True)
    os.makedirs(destination + "val", exist_ok=True)
    os.makedirs(destination + "test", exist_ok=True)
    
    train_end = int(TRAIN_RATIO*number_files)
    eval_end = train_end + int(EVAL_RATIO*number_files)
    
    for i, file in enumerate(files):
        if i < train_end:
            dst = "{}/train/{}".format(destination, os.path.basename(file))
        elif i < eval_end:
            dst = "{}/val/{}".format(destination, os.path.basename(file))
        else:
            dst = "{}/test/{}".format(destination, os.path.basename(file))
        os.rename(file, dst)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Split data into training / validation / testing')
    parser.add_argument('--source', required=True,
                        help='source data directory')
    parser.add_argument('--destination', required=True,
                        help='destination data directory')
    args = parser.parse_args()
    
    total = TRAIN_RATIO + EVAL_RATIO + TEST_RATIO
    
    assert total == 1.0, f"the sum of probability must be 1.0, got: {total}"

    logger = get_module_logger(__name__)
    logger.info('Creating splits...')
    split(args.source, args.destination)