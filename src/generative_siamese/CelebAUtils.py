# coding=utf-8
"""Utility methods used for loading and evaluating the CelebA dataset."""
import os
from zipfile import ZipFile
import gdown


def download_dataset(path):
    """Download and unzips the CelebA dataset to the specified path."""
    print("downloading the CelebA dataset...")
    gdown.download(
        "https://drive.google.com/uc?id=0B7EVK8r0v71pZjFTYXZWM3FlRnM",
        os.path.join(path, 'CelebA.zip'), True)
    gdown.download(
        "https://drive.google.com/uc?id=1_ee_0u7vcNLOfNLegJRHmolfH5ICW-XS",
        os.path.join(path, 'ids.txt'), True)
    gdown.download(
        "https://drive.google.com/uc?id=0B7EVK8r0v71pblRyaVFSWGxPY0U",
        os.path.join(path, 'attr.txt'), True)
    print("done")


def unzip_dataset(source_path):
    """Unzip the CelebA dataset to the path CelebA_unzipped."""
    with ZipFile(os.path.join(source_path, 'CelebA.zip')) as zip_ref:
        print("unzipping the CelebA dataset...")
        zip_ref.extractall(os.path.join(source_path, 'CelebA_unzipped'))
    print("done")


def get_dataset(path, dataset):
    """Download and extract the dataset."""
    found_unzipped = os.path.exists(os.path.join(path, dataset + "_unzipped"))
    found_zipped = os.path.isfile(os.path.join(path, dataset + ".zip"))

    if not (found_zipped or found_unzipped):
        download_dataset(path)
    else:
        if found_zipped:
            print("Found the zipped dataset")

    if not found_unzipped:
        unzip_dataset(path)
    else:
        print("Found the unzipped dataset")
