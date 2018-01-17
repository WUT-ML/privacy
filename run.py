#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Main experiment pipeline."""

import glob
import hashlib
import os
import shutil
import time
import urllib.request
import zipfile

import luigi


NIST_URL = 'https://s3.amazonaws.com/nist-srd/SD4/NISTSpecialDatabase4GrayScaleImagesofFIGS.zip'
NIST_SHA = '4db6a8f3f9dc14c504180cbf67cdf35167a109280f121c901be37a80ac13c449'


def get_hash(path):
    """Return a SHA256 hash of a given file."""
    sha = hashlib.sha256()
    BLOCKSIZE = 65536

    try:
        with open(path, 'rb') as file:
            buffer = file.read(BLOCKSIZE)
            while len(buffer) > 0:
                sha.update(buffer)
                buffer = file.read(BLOCKSIZE)
    except:
        return None

    return sha.hexdigest()


class PipelineException(Exception):
    """Custom workflow exception."""


class CleanableTask(luigi.Task):
    """A task with cleaning functionality."""

    def clean(self):
        self.output().remove()


class DownloadDataset(CleanableTask):

    """Download the NIST FIGS dataset."""

    def output(self):
        """Dataset ZIP archive."""
        return luigi.LocalTarget('data/NIST-FIGS.zip')

    def run(self):
        """Download dataset ZIP archive."""
        with self.output().temporary_path() as tmp_path:
            urllib.request.urlretrieve(NIST_URL, tmp_path)

            if get_hash(tmp_path) != NIST_SHA:
                os.remove(tmp_path)
                raise PipelineException('Dataset checksum verification failed.')


class ExtractDataset(CleanableTask):

    """Extract the NIST FIGS dataset."""

    def requires(self):
        return DownloadDataset()

    def output(self):
        """Dataset directory."""
        return luigi.LocalTarget('data/NISTSpecialDatabase4GrayScaleImagesofFIGS')

    def run(self):
        """Extract and verify the dataset."""
        with self.output().temporary_path() as tmp_path:
            os.mkdir(tmp_path)

            with zipfile.ZipFile(self.input().path, 'r') as zip:
                zip.extractall(tmp_path)

            tmp_path += '/'
            inside_dir = tmp_path + 'NISTSpecialDatabase4GrayScaleImagesofFIGS/'
            for filename in os.listdir(inside_dir):
                shutil.move(inside_dir + filename, tmp_path + filename)
            os.rmdir(inside_dir)

            if len(glob.glob(tmp_path + '**/*.txt', recursive=True)) != 4000:
                raise PipelineException('Dataset image count verification failed.')


class Dataset(CleanableTask):

    """Prepare the NIST FIGS dataset."""

    def requires(self):
        return ExtractDataset()

    def output(self):
        return self.input()


if __name__ == '__main__':
    luigi.run()
