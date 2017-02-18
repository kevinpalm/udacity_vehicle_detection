from os import walk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from preprocess import HogPreprocessor

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


def summarize_classes(df):
    """ Check and plot the training examples class proportions """

    # Prepare a summary
    class_counts = df.groupby("vehicle").count()
    class_counts.columns = ["Frequency"]

    # Plot and save
    class_counts.plot(kind="bar", title="Training Data Class Frequencies (Non-vehicle vs Vehicle)", legend=False)
    plt.savefig("../output_images/class_frequencies.jpg")
    plt.clf()


def stratified_subsample(df, n=100):
    """ Retrive a subsample with an equal proportion of training classes """

    # Sample each class
    sample_v = df[df["vehicle"] == 1.0].sample(n=int(n/2))
    sample_nv = df[df["vehicle"] == 0.0].sample(n=int(n / 2))

    # Append together
    output = sample_v.append(sample_nv)

    return output


def read_images(df):
    """ Given an ordered dataframe, read all the image files and return an ordered numpy array"""

    # Read the images
    images = np.array([imread(x) for x in df["file"].values])

    return images


def plot_preprocessor(image, preprocessor):

    # Make a visualization of the preprocessor transform
    visualization = preprocessor[1].make_visualization(image)

    # Plot the example
    fig = plt.figure()
    plt.subplot(121)
    print(image.shape)
    plt.imshow(image, cmap='gray')
    plt.title('Example Car Image')
    plt.subplot(122)
    plt.imshow(visualization, cmap='gray')
    plt.title('Example {}'.format(preprocessor[0]))
    plt.savefig("../output_images/{}.jpg".format(preprocessor[0].lower().replace(" ", "_")))
    plt.clf()



def main():
    """ Graph the training class proportions and save pickle training data files"""

    # # Retreive the file locations
    # training_file_locations = get_image_locations()
    #
    # # Format the file locations to a pandas dataframe
    # training_df = format_image_locations(training_file_locations)
    #
    # # Graph the class proportions
    # summarize_classes(training_df)
    #
    # # Read in the images
    # images = read_images(training_df)
    #
    # # Pickle the formatted data
    # training_df.to_pickle("../model_saves/training_labels.pkl")
    # images.dump("../model_saves/training_data.pkl")

    # Plot some examples of preprocessing steps
    example = imread("../input_images/vehicles/GTI_MiddleClose/image0122.png")
    preprocessors = [("Gray HOG", HogPreprocessor(color_channel="gray"))]
    for preprocessor in preprocessors:
        plot_preprocessor(example, preprocessor)


if __name__ == '__main__':
    main()