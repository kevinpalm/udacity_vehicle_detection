from preprocess import HogPreprocessor
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score


def test_hog_channels(x, y):
    """ Run through a series of default linear SVM tests using different color channels before the hog transform"""

    print("Starting the hog color channel tests...")

    # Split the dataset up
    train_x, test_x, train_y, test_y = train_test_split(x, y)

    # Define the estimator
    estimator = LinearSVC()

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

        # Preprocess and train
        processed_train_x = hog[1].transform(train_x)
        estimator.fit(processed_train_x, train_y)

        # Preprocess the test data, report a score
        processed_test_x = hog[1].transform(test_x)
        print("Using {0} as the color channel,"
              "the accuracy was {1}".format(hog[0], accuracy_score(test_y, estimator.predict(processed_test_x))))

    """
    Output:
    Using red as the color channel,the accuracy was 0.8673423423423423
    Using green as the color channel,the accuracy was 0.8673423423423423
    Using blue as the color channel,the accuracy was 0.8673423423423423
    Using gray as the color channel,the accuracy was 0.8788288288288288
    Using saturation as the color channel,the accuracy was 0.81509009009009
    Using lightness as the color channel,the accuracy was 0.8835585585585586
    Using hue as the color channel,the accuracy was 0.8286036036036036
    """


def main():

    # Load in the data
    labels = pd.read_pickle("../model_saves/training_labels.pkl")
    data = np.load("../model_saves/training_data.pkl")


if __name__ == '__main__':
    main()
