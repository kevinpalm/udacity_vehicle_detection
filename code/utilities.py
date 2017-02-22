from os import walk
import pandas as pd
import numpy as np
from scipy.misc import imread


def get_image_locations():
    """ Get a list of all the training images that are going to be used for training data """

    # Create a placeholder list
    locations = []

    # Iterate through the directories
    for dirpath, dirnames, filenames in walk("../input_images"):

        # Iterate through the files
        for file in filenames:

            # Only append if the file is an png
            if file[-4:] == ".png":
                locations.append(dirpath + "/" + file)

    return locations


def format_image_locations(locations):
    """Prepare a pandas dataframe containing the image locations, including a column for which type of image """

    # Create a placeholder dataframe
    df = pd.DataFrame()

    # Store the location
    df["file"] = locations

    # Mark vehicles as 1s, non-vehicles as 0
    df["vehicle"] = 0.0
    df["vehicle"][df["file"].str[:25] == "../input_images/vehicles/"] = 1.0

    return df


def stratified_subsample(df, n=100):
    """ Retrive a subsample with an equal proportion of training classes """

    # Sample each class
    sample_v = df[df["vehicle"] == 1.0].sample(n=int(n / 2))
    sample_nv = df[df["vehicle"] == 0.0].sample(n=int(n / 2))

    # Append together
    output = sample_v.append(sample_nv)

    return output


def read_images(df):
    """ Given an ordered dataframe, read all the image files and return an ordered numpy array"""

    # Read the images
    images = np.array([imread(x) for x in df["file"].values])

    return images


