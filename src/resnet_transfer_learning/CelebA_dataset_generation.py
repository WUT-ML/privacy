# coding=utf-8
"""Utility methods used for splitting the CelebA dataset by attributes for transfer learning."""
import os
import numpy as np
import glob
from shutil import copyfile


def __get_array_of_attributes():
    """Process attr.txt file of CelebA dataset. Only valid for the default path."""
    attr_file = open(os.path.join('data', 'CelebA', 'attr.txt'), 'r')
    lines = attr_file.read().split("\n")
    attr_file.close()

    df = []
    for line in lines[1: -1]:
        line = line.split(" ")
        line = list(filter(None, line))
        assert len(line) == 41

        line[0] = int(line[0][:line[0].find('.')])
        line_as_ints = list(map(int, line))
        df.append(line_as_ints)

    return np.array(df)


def split_images_into_train_and_val(path_source, path_destination, attribute,
                                    distortion_constraint, validation_percentage):
    """Split set of images into training and validation set and divide into two categories.

    Keyword arguments:
    path_source -- full path to the dataset to be split
    path_destination -- full path to the place where the split dataset is to be saved
    attribute -- the attribute according to which the category is determined, in rage: (1, 40)
    distortion_constraint -- distortion constraint of the set to be processed. Set to 0 if
    original set is to be processed.
    validation_percentage -- int representing how much % of the dataset should be in validation set
    """
    path_with_attribute = os.path.join(path_destination, 'attribute_' + str(attribute),
                                       'distortion_' + str(distortion_constraint))
    if not os.path.exists(path_with_attribute):
        os.makedirs(path_with_attribute)
    filenames_generator = glob.iglob(os.path.join(path_source, '**', '*.jpg'), recursive=True)
    filenames_with_paths = list(filenames_generator)
    filenames_strings = np.array([name[name.rfind(os.sep) + 1: name.rfind('.')] for name in
                                  filenames_with_paths])
    filenames = filenames_strings.astype(int)

    attributes = __get_array_of_attributes()
    ids_with_attribute = np.where(attributes[:, attribute] == 1)[0] + 1
    ids_without_attribute = np.where(attributes[:, attribute] == -1)[0] + 1
    indices_of_paths_with_attribute = np.nonzero(np.in1d(filenames, ids_with_attribute))[0]
    indices_of_paths_without_attribute = np.nonzero(np.in1d(filenames, ids_without_attribute))[0]
    paths_with_attribute = [filenames_with_paths[i] for i in indices_of_paths_with_attribute]
    paths_without_attribute = [filenames_with_paths[i] for i in indices_of_paths_without_attribute]

    number_of_val_images_with_attribute = round(validation_percentage / 100 *
                                                len(paths_with_attribute))
    number_of_val_images_without_attribute = round(validation_percentage / 100 *
                                                   len(paths_without_attribute))
    paths_val_with_attribute, paths_train_with_attribute = \
        paths_with_attribute[:number_of_val_images_with_attribute], paths_with_attribute[
                                                           number_of_val_images_with_attribute:]
    paths_val_without_attribute, paths_train_without_attribute = paths_without_attribute[
        :number_of_val_images_without_attribute], paths_without_attribute[
                                                         number_of_val_images_without_attribute:]

    # copy images into train/val folders split into with/without attribute
    for with_or_without in ['with', 'without']:
        if with_or_without == 'with':
            train_paths, val_paths = paths_train_with_attribute, paths_val_with_attribute
        else:
            train_paths, val_paths = paths_train_without_attribute, paths_val_without_attribute

        for train_or_val in ['train', 'val']:
            if train_or_val == 'train':
                paths_to_images = train_paths
            else:
                paths_to_images = val_paths
            for path_to_image in paths_to_images:
                dst_dir = os.path.join(path_with_attribute, train_or_val, with_or_without)
                if not os.path.exists(dst_dir):
                    os.makedirs(dst_dir)
                dst_path = os.path.join(dst_dir, path_to_image[path_to_image.rfind(os.sep) + 1:])
                copyfile(path_to_image, dst_path)
