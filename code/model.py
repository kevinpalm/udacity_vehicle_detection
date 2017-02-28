from preprocess import HogPreprocessor, HistoPreprocessor, CannyBinPreprocessor
from boxes import *
import cv2
from tests import run_local_test
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFwe, SelectKBest
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import RidgeClassifier, Perceptron


# def get_pipelines():
#     """ returns the untrained preprocessors and estimator """
#
#     # Define the preprocessors to use
#     preprocessors = [Pipeline([("gray_hog", HogPreprocessor(color_channel="gray")),
#                                ("scale1", MinMaxScaler()),
#                                ("pca", PCA(svd_solver="randomized")),
#                                ("scale2", MinMaxScaler()),
#                                ("select", SelectFwe(alpha=0.01))]),
#                      Pipeline([("red_histo", HistoPreprocessor(color_channel="red")),
#                                ("scale1", MinMaxScaler()),
#                                ("pca", PCA(svd_solver="randomized")),
#                                ("scale2", MinMaxScaler()),
#                                ("select", SelectFwe(alpha=0.01))]),
#                      Pipeline([("green_histo", HistoPreprocessor(color_channel="green")),
#                                ("scale1", MinMaxScaler()),
#                                ("pca", PCA(svd_solver="randomized")),
#                                ("scale2", MinMaxScaler()),
#                                ("select", SelectFwe(alpha=0.01))]),
#                      Pipeline([("blue_histo", HistoPreprocessor(color_channel="blue")),
#                                ("scale1", MinMaxScaler()),
#                                ("pca", PCA(svd_solver="randomized")),
#                                ("scale2", MinMaxScaler()),
#                                ("select", SelectFwe(alpha=0.01))]),
#                      Pipeline([("gray_canny_bin", CannyBinPreprocessor(color_channel="gray")),
#                                ("scale1", MinMaxScaler()),
#                                ("pca", PCA(svd_solver="randomized")),
#                                ("scale2", MinMaxScaler()),
#                                ("select", SelectFwe(alpha=0.01))])]
#
#     estimator = Pipeline([("pca", PCA(svd_solver="randomized")),
#                           ("scale", MinMaxScaler()),
#                           ("select", SelectKBest(k=32)),
#                           ("gb", GradientBoostingClassifier(subsample=0.5, n_estimators=1000, learning_rate=0.01))])
#
#     return preprocessors, estimator

def get_pipelines():
    """ returns the untrained preprocessors and estimator """

    # Define the preprocessors to use
    preprocessors = [Pipeline([("gray_hog", HogPreprocessor(color_channel="gray")),
                               ("scale1", MinMaxScaler())]),
                     Pipeline([("red_histo", HistoPreprocessor(color_channel="red")),
                               ("scale1", MinMaxScaler())]),
                     Pipeline([("green_histo", HistoPreprocessor(color_channel="green")),
                               ("scale1", MinMaxScaler())]),
                     Pipeline([("blue_histo", HistoPreprocessor(color_channel="blue")),
                               ("scale1", MinMaxScaler())]),
                     Pipeline([("gray_canny_bin", CannyBinPreprocessor(color_channel="gray")),
                               ("scale1", MinMaxScaler())])]

    estimator = VotingClassifier([("perceptron", Perceptron()),
                                  ("ridge", RidgeClassifier(alpha=1000)),
                                  ("svm", LinearSVC())])

    return preprocessors, estimator

def test_pipelines():
    """ load the training data, get the pipelines, and run a train-test split test"""

    # Load in the data
    labels = pd.read_pickle("../model_saves/training_labels.pkl")
    data = np.load("../model_saves/training_data.pkl")

    # Get the preprocessor and estimator configurations
    preprocessors, estimator = get_pipelines()

    # Run a local test
    run_local_test(preprocessors, estimator, data, labels["vehicle"].values)


def scratch_train_save():
    """ Train the preprocessors and estimator from scratch, then save them for future use"""

    # Try and load the data
    try:
        labels = pd.read_pickle("../model_saves/training_labels.pkl")
        data = np.load("../model_saves/training_data.pkl")
    except:
        raise ValueError("The data pickles don't seem to be loading. Did you create them by running"
                         " utilities.py or full_run.py?")

    # Get the preprocessors and estimator
    preprocessors, estimator = get_pipelines()

    # Fit the preprocessors

    # Train each of the preprocessors
    trained_preprocessors = []
    for preprocessor in preprocessors:
        trained_preprocessors.append(preprocessor.fit(data, labels["vehicle"].values))

    # Transform the training data
    preprocessed_data = np.concatenate([x.transform(data) for x in trained_preprocessors], axis=1)

    # Train the estimator
    estimator.fit(preprocessed_data, labels["vehicle"].values)

    # Save the preprocessors
    names = ["gray_hog", "red_histo", "green_histo", "blue_histo", "gray_canny_bin"]
    for preprocessor, name in zip(trained_preprocessors, names):
        joblib.dump(preprocessor, '../model_saves/preprocessor_{}.pkl'.format(name))

    # Save the model
    joblib.dump(estimator, "../model_saves/estimator.pkl")

    return trained_preprocessors, estimator


def load_pretrained(pre_locs, est_loc):
    """ Load the trained preprocessors and estimator"""

    # Load the preprocessors
    preprocessors = [joblib.load(x) for x in pre_locs]

    # Load the estimator
    estimator = joblib.load(est_loc)

    return preprocessors, estimator


def make_estimates(X, preprocessors, estimator, proba=False):
    """ Make predictions """

    # Apply the preprocessing
    preprocessed_X = np.concatenate([x.transform(X) for x in preprocessors], axis=1)

    # Make predictions
    if proba is True:
        try:
            predicts = estimator.predict_proba(preprocessed_X)
        except:
            predicts = estimator.predict(preprocessed_X)
    else:
        predicts = estimator.predict(preprocessed_X)

    return predicts

def search_windows(img, windows, preprocessors, estimator):
    """ Pass an image and the list of windows to be searched (output of slide_windows()) """

    # Create an empty list to receive positive detection windows
    on_windows = []


    # Get the images
    images = np.array([cv2.resize(img[window[0][1]:window[1][1],
                                  window[0][0]:window[1][0]], (64, 64)) for window in windows])

    # Get the predictions
    predicts = make_estimates(images, preprocessors, estimator)

    # Build the active window list
    for prediction, window in zip(predicts, windows):

        # If positive then save the window
        if prediction == 1:
            on_windows.append(window)

    # Return windows for positive detections
    return on_windows


def get_search_window_list():
    """ Get a list of search windows appropriate to a 1280 x 720 image """

    # Define the search parameters
    x_start_stops = [(0, 1280), (16, 1264), (0, 1280)]
    y_start_stops = [(400, 656), (400, 608), (400, 528)]
    xy_windows = [(128, 128), (96, 96), (64, 64)]
    xy_overlaps = [(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]

    # Create a placeholder list
    search_list = []

    # Run though each set of windows and append
    for xss, yss, w, o in zip(x_start_stops, y_start_stops, xy_windows, xy_overlaps):
        search_list.extend(slide_window(x_start_stop=xss, y_start_stop=yss, xy_window=w, xy_overlap=o))

    return search_list

def load_model():

    # Try and load already created preprocessors and estimator
    try:
        pre_locs = ["../model_saves/preprocessor_gray_hog.pkl",
                    "../model_saves/preprocessor_red_histo.pkl", "../model_saves/preprocessor_green_histo.pkl",
                    "../model_saves/preprocessor_blue_histo.pkl", "../model_saves/preprocessor_gray_canny_bin.pkl"]
        est_loc = "../model_saves/estimator.pkl"
        preprocessors, estimator = load_pretrained(pre_locs, est_loc)

    # If the load didn't work, create them from scratch
    except:
        print("Model load failed, creating from scratch...")
        preprocessors, estimator = scratch_train_save()

    return preprocessors, estimator


class BoxImageClassifer:
    """ Helper class for actually making predictions"""

    def __init__(self, preprocessors, estimator, windows):

        self.preprocessors = preprocessors
        self.estimator = estimator
        self.windows = windows
        self.past_frames = None

    def predit(self, image):

        # Search for active windows
        box_list = search_windows(image, self.windows, self.preprocessors, self.estimator)

        # Create a heatmap
        heat = np.zeros(image.shape[:2])

        # Add heat to each box in box list
        heat = add_heat(heat, box_list)

        # # Apply threshold to help remove false positives
        # heat = apply_threshold(heat, 1)

        # Check if this is the first frame
        if self.past_frames is None:
            self.past_frames = heat

        # Otherwise, scale down the past frames and include in the calculation
        else:
            heat = heat + (self.past_frames*0.8)
            self.past_frames = heat

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 2.8)

        # Visualize the heatmap when displaying
        heatmap = np.clip(heat, 0, 255)

        # Find final boxes from heatmap using label function
        labels = label(heatmap)
        draw_img = draw_labeled_bboxes(np.copy(image), labels)

        return draw_img


def main():
    """ Model and estimate all the project materials """

    # Run a local test with the current configuration
    test_pipelines()

    # # Load the model
    # preprocessors, estimator = load_model()
    #
    # # Load the search list
    # search_list = get_search_window_list()
    #
    # # Define the classifer
    # clf = BoxImageClassifer(preprocessors, estimator, search_list)
    #
    # # Highlight the video clips
    # # "../project_video.mp4",
    # for cliploc in ["../project_video.mp4"]:
    #     clip = VideoFileClip(cliploc).fl_image(clf.predit)
    #     clip.write_videofile(cliploc.replace("../", "../processed_"), audio=False)


if __name__ == '__main__':
    main()
