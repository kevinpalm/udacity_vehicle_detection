from preprocess import HogPreprocessor, HistoPreprocessor, CannyBinPreprocessor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler


def test_hog_channels(x, y):
    """ Run through a series of default linear SVM tests using different color channels before the hog transform"""

    print("Starting the hog color channel tests...")

    # Split the dataset up
    train_x, test_x, train_y, test_y = train_test_split(x, y)

    # Define the preprocessors to test
    hogs = [("red", HogPreprocessor(color_channel="red")),
            ("green", HogPreprocessor(color_channel="green")),
            ("blue", HogPreprocessor(color_channel="blue")),
            ("gray", HogPreprocessor(color_channel="gray")),
            ("saturation", HogPreprocessor(color_channel="saturation")),
            ("lightness", HogPreprocessor(color_channel="lightness")),
            ("hue", HogPreprocessor(color_channel="hue"))]

    # Iterate through the preprocessors
    for hog in hogs:

        # Define and train estimator
        estimator = Pipeline([hog, ("svc", LinearSVC())]).fit(train_x, train_y)

        # Report a score
        print("Using {0} as the color channel,"
              "the accuracy was {1}".format(hog[0], accuracy_score(test_y, estimator.predict(test_x))))

    """
    Example output:
    Using red as the color channel,the accuracy was 0.8626126126126126
    Using green as the color channel,the accuracy was 0.8804054054054054
    Using blue as the color channel,the accuracy was 0.8677927927927928
    Using gray as the color channel,the accuracy was 0.8786036036036036
    Using saturation as the color channel,the accuracy was 0.8051801801801802
    Using lightness as the color channel,the accuracy was 0.8801801801801802
    Using hue as the color channel,the accuracy was 0.8326576576576576
    """


def test_hist_channels(x, y):
    """ Run through a series of default linear SVM tests using different color channels before the histogram summary"""

    print("Starting the histogram color channel tests...")

    # Split the dataset up
    train_x, test_x, train_y, test_y = train_test_split(x, y)

    # Define the preprocessors to test
    hogs = [("red", HistoPreprocessor(color_channel="red")),
            ("green", HistoPreprocessor(color_channel="green")),
            ("blue", HistoPreprocessor(color_channel="blue")),
            ("gray", HistoPreprocessor(color_channel="gray")),
            ("saturation", HistoPreprocessor(color_channel="saturation")),
            ("lightness", HistoPreprocessor(color_channel="lightness")),
            ("hue", HistoPreprocessor(color_channel="hue"))]

    # Iterate through the preprocessors
    for hog in hogs:

        # Define and train estimator
        estimator = Pipeline([hog, ("svc", LinearSVC())]).fit(train_x, train_y)

        # Report a score
        print("Using {0} as the color channel,"
              "the accuracy was {1}".format(hog[0], accuracy_score(test_y, estimator.predict(test_x))))

    """
    Example output:
    Using red as the color channel,the accuracy was 0.8342342342342343
    Using green as the color channel,the accuracy was 0.8387387387387387
    Using blue as the color channel,the accuracy was 0.779054054054054
    Using gray as the color channel,the accuracy was 0.8144144144144144
    Using saturation as the color channel,the accuracy was 0.6939189189189189
    Using lightness as the color channel,the accuracy was 0.7065315315315316
    Using hue as the color channel,the accuracy was 0.8029279279279279
    """

def test_canny_bins(x, y):
    """ Try using the canny bins as the only features, just for an idea of their descriptiveness """

    print("Starting the canny bin test...")

    # Split the dataset up
    train_x, test_x, train_y, test_y = train_test_split(x, y)

    # Define and train estimator
    estimator = Pipeline([("cb", CannyBinPreprocessor()),
                          ("scale", MinMaxScaler()),
                          ("svc", LinearSVC())]).fit(train_x, train_y)

    # Report a score
    print("The accuracy was {}".format(accuracy_score(test_y, estimator.predict(test_x))))


def run_local_test(preprocessors, estimator, x, y):
    """ Run a local test by splitting the data to help with tuning"""

    print("Starting a local test...")

    # Split the dataset up
    train_x, test_x, train_y, test_y = train_test_split(x, y)

    # Train each of the preprocessors
    trained_preprocessors = []
    for preprocessor in preprocessors:
        trained_preprocessors.append(preprocessor.fit(train_x, train_y))

    # Apply the preprocessing
    preprocessed_train_x = np.concatenate([x.transform(train_x) for x in trained_preprocessors], axis=1)
    preprocessed_test_x = np.concatenate([x.transform(test_x) for x in trained_preprocessors], axis=1)

    print("Fitting the estimator to {} final features...".format(preprocessed_train_x.shape[1]))

    # Train the estimator
    estimator.fit(preprocessed_train_x, train_y)

    # Report a score
    print("The local test AUC was {}".format(roc_auc_score(test_y, estimator.predict(preprocessed_test_x))))


def main():

    # Load in the data
    labels = pd.read_pickle("../model_saves/training_labels.pkl")
    data = np.load("../model_saves/training_data.pkl")

    # Run some tests to determine which color channel to use for HOG features
    test_hog_channels(data, labels["vehicle"].values)

    # Run some tests to determine which color channel to use for hist features
    test_hist_channels(data, labels["vehicle"].values)

    # Try using canny bins as the only features
    test_canny_bins(data, labels["vehicle"].values)


if __name__ == '__main__':
    main()