# coding=utf-8
"""Utility methods used for loading the FERG dataset."""
import requests
import os
from zipfile import ZipFile
import csv
import glob


def download_dataset(path):
    """Download and unzips the FERG dataset to the specified path."""
    url = "http://grail.cs.washington.edu/projects/deepexpr/FERG_DB_256.zip"
    print("downloading the FERG dataset...")
    r = requests.get(url, allow_redirects=True)
    open(os.path.join(path, 'FERG.zip'), 'wb').write(r.content)
    print("done")


def unzip_dataset(source_path):
    """Unzip the FERG dataset to the path FERG_unzipped."""
    print("FERG-DB dataset requires you to fill out the agreement form to get access."
          "It can be found here: "
          "https://grail.cs.washington.edu/projects/deepexpr/ferg-db.html")
    password = input("password to the FERG dataset: ")
    with ZipFile(os.path.join(source_path, 'FERG.zip')) as zip_ref:
        print("unzipping the FERG dataset...")
        zip_ref.extractall(os.path.join(source_path, 'FERG_unzipped'), pwd=bytes(password, 'utf-8'))
    # remove a faulty image
    os.remove(os.path.join(
        source_path, 'FERG_unzipped', 'FERG_DB_256', 'bonnie', 'bonnie_surprise',
        'bonnie_surprise_1389.png'))
    print("done")


def get_dataset(path, dataset):
    """Download and extract the dataset."""
    found_unzipped = os.path.exists(os.path.join(path, dataset + "_unzipped"))
    found_zipped = os.path.isfile(os.path.join(path, dataset + ".zip"))
    found_csv = os.path.isfile(os.path.join(path, "images.csv"))

    if not (found_zipped or found_unzipped):
        download_dataset(path)
    else:
        if found_zipped:
            print("Found the zipped dataset")

    if not found_unzipped:
        unzip_dataset(path)
    else:
        print("Found the unzipped dataset")

    if not found_csv:
        generate_csv(path)


def generate_csv(path):
    """Generate csv file with list of paths to all the images."""
    list_of_faces = glob.glob(os.path.join(path, '**', '*.png'), recursive=True)
    names_in_desired_order = ['mery', 'aia', 'malcolm', 'bonnie', 'ray', 'jules']
    emotions_alphabetically = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
    ordered_list_of_faces = []
    for name in names_in_desired_order:
        ordered_list_of_faces.append([x for x in list_of_faces if name in x])

    with open(os.path.join(path, 'images.csv'), 'w') as csvfile:
        filepaths_writer = csv.writer(csvfile, quotechar='|', quoting=csv.QUOTE_MINIMAL)
        for identity in ordered_list_of_faces:
            for sample in identity:
                for emotion_index in range(len(emotions_alphabetically)):
                    if emotions_alphabetically[emotion_index] in sample:
                        filepaths_writer.writerow([sample, str(emotion_index)])
