from os import walk
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from preprocess import HogPreprocessor, HistoPreprocessor, CannyBinPreprocessor
from model import load_pretrained, make_estimates

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
    visualization, type = preprocessor[1].make_visualization(image)

    # Visualize based on the type
    if type == "hog":
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Example Car Image')
        ax = plt.subplot(122)
        plt.imshow(visualization, cmap='gray')
        plt.title('Example {}'.format(preprocessor[0]))

    elif type == "histo":
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(image, cmap='gray')
        plt.title('Example Car Image')
        ax = plt.subplot(122)
        ax.bar(visualization[0], visualization[1], width=1/32)
        ax.set_xlim([0, 1])
        plt.title('Example {}'.format(preprocessor[0]))

    elif type == "canny":
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(image, cmap='gray')
        plt.title('Example Car Image')
        ax = plt.subplot(132)
        ax.bar(visualization[0][0], visualization[0][1], width=0.02)
        plt.title('X-axis Mean Canny Histogram')
        ax2 = plt.subplot(133)
        ax2.barh(visualization[1][0], visualization[1][1], height=0.02)
        plt.title('Y-axis Mean Canny Histogram'.format(preprocessor[0]))

    # Finish and save
    plt.savefig("../output_images/{}.jpg".format(preprocessor[0].lower().replace(" ", "_")))
    plt.clf()

def estimator_sample_results():

    # Load the model
    pre_locs = ["../model_saves/preprocessor_lightness_hog.pkl", "../model_saves/preprocessor_red_histo.pkl",
                "../model_saves/preprocessor_green_histo.pkl", "../model_saves/preprocessor_blue_histo.pkl",
                "../model_saves/preprocessor_canny_bin.pkl"]
    est_loc = "../model_saves/estimator.pkl"
    preprocessors, estimator = load_pretrained(pre_locs, est_loc)

    # Load the labels
    labels = pd.read_pickle("../model_saves/training_labels.pkl")
    cars = labels[labels["vehicle"] == 1]
    noncars = labels[labels["vehicle"] == 0]
    samples = cars.sample(n=3).append(noncars.sample(n=3))["file"].tolist()

    # Draw and estimate each sample
    fig = plt.figure(figsize=(12, 3))
    fig.suptitle("Example Estimated Likelihood of Vehicles:", fontsize=14)
    for n in range(6):

        # Save and estimate the image
        image = imread(samples[n])
        imagea = np.array([image])
        predict = np.round(make_estimates(imagea, preprocessors, estimator, proba=True), decimals=3)
        plt.subplot(1, 6, n+1)
        plt.imshow(image, cmap="gray")
        plt.title(str(predict[0][1]))

    plt.tight_layout()
    plt.savefig("../output_images/sample_training_estimates.jpg")
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
    #
    # # Plot some examples of preprocessing steps
    # example = imread("../input_images/vehicles/GTI_MiddleClose/image0122.png")
    # preprocessors = [("Lightness HOG", HogPreprocessor(color_channel="lightness")),
    #                  ("Red Histogram", HistoPreprocessor(color_channel="red")),
    #                  ("Green Histgram", HistoPreprocessor(color_channel="green")),
    #                  ("Blue Histogram", HistoPreprocessor(color_channel="blue")),
    #                  ("Canny Bin Histogram", CannyBinPreprocessor())]
    # for preprocessor in preprocessors:
    #     plot_preprocessor(example, preprocessor)
    #
    # # Make some examples of the estimator performance
    # estimator_sample_results()


if __name__ == '__main__':
    main()