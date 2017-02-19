from sklearn.base import TransformerMixin
from skimage.feature import hog, canny
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Preprocessor(TransformerMixin):
    """ Meta class for this project's prepocessors, handling scale and color transforms"""

    def __init__(self, color_channel="gray"):

        # Run TransformerMixin init as well
        super().__init__()

        # Store which color channel to analyze
        if color_channel in ['gray', 'grey', 'red', 'green', 'blue', 'hue','saturation', 'lightness']:
            self.color_channel = color_channel
        else:
            raise ValueError("An invalid argument was selected for color_channel. Valid arguments are:"
                             " ['gray', 'red', 'green', 'blue', 'hue','saturation', 'lightness']")


    def color_transform(self, input):
        # Reduce the input to a single color channel of the preprocessor's specified type

        # Gray
        if self.color_channel == "gray" or self.color_channel == "grey":
            output = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
        # Red
        elif self.color_channel == "red":
            output = input[:,:,0]
        # Green
        elif self.color_channel == "green":
            output = input[:,:,1]
        # Blue
        elif self.color_channel == "blue":
            output = input[:,:,2]
        # Hue
        elif self.color_channel == "hue":
            output = cv2.cvtColor(input, cv2.COLOR_RGB2HLS)[:,:,0]
        # Saturation
        elif self.color_channel == "saturation":
            output = cv2.cvtColor(input, cv2.COLOR_RGB2HLS)[:,:,2]
        # Hue
        elif self.color_channel == "lightness":
            output = cv2.cvtColor(input, cv2.COLOR_RGB2HLS)[:,:,1]
        else:
            raise ValueError("An invalid argument was selected for color_channel. Valid arguments are:"
                             " ['gray', 'red', 'green', 'blue', 'hue','saturation', 'lightness']")

        return output


    def ensure_scale(self, input):
        """ Make sure that the input numpy array is scaled to be between 0 and 1"""

        # Check if out of scale, if so fix it
        if input.max() > 1.0:
            output = input / 255.0
        else:
            output = input

        # Check the scale again
        if output.max() > 1.0:
            raise ValueError("Input images must be 8-bit (255 values) or pre-scaled floats within (0.0, 1.0).")

        return output


    def fit(self, X, y=None):
        """ Just a placeholder for the fit method if this is added to a sklearn pipeline"""

        return self


    def pipeline(self, X):
        """ Another placeholder which will be replaced by the subclass"""

        return X


    def transform(self, X):
        """ Wrapper function for pipeline, which will handling single images and arrays of images"""

        # Check to make sure this an array of multiple images, and if so apply to each image
        if len(X.shape) == 4:
            X = np.array([self.pipeline(x) for x in X])

        # Check if it's a single image
        elif len(X.shape) == 3:
            X = self.pipeline(X)

        # Raise error if it's neither shape
        else:
            raise ValueError("The input doesn't appear to be valid 3 channel image(s).")

        return X


class HogPreprocessor(Preprocessor):
    """ Preprocessor for Hog features """

    def __init__(self, color_channel="gray", pix_per_cell=8, cell_per_block=2, orient=2):

        # Run Preprocessor init as well
        super().__init__(color_channel=color_channel)

        # Store arguments
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.orient = orient

    def pipeline(self, X):
        """ The list of opperations to apply per each image"""

        # Convert the color
        X = self.color_transform(X)

        # Ensure the scale
        X = self.ensure_scale(X)

        # Apply the hog transform
        X = hog(X, orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                       cells_per_block=(self.cell_per_block, self.cell_per_block), feature_vector=False)

        # Flatten the output
        X = X.flatten()

        return X

    def make_visualization(self, X):
        """ Return a Hog visualization from a single example image"""

        # Convert the color
        X = self.color_transform(X)

        # Ensure the scale
        X = self.ensure_scale(X)

        # Apply the hog transform
        X, visualization = hog(X, orientations=self.orient, pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                       cells_per_block=(self.cell_per_block, self.cell_per_block), visualise=True, feature_vector=False)

        return visualization, "hog"


class HistoPreprocessor(Preprocessor):
    """ Preprocessor for creating histogram summary features """

    def __init__(self, color_channel="gray", nbins=32):

        # Run Preprocessor init as well
        super().__init__(color_channel=color_channel)

        # Store arguments
        self.nbins = nbins

    def pipeline(self, X):
        """ The list of opperations to apply per each image"""

        # Convert the color
        X = self.color_transform(X)

        # Ensure the scale
        X = self.ensure_scale(X)

        # Prepare a histogram
        X = np.float64(np.histogram(X, bins=self.nbins, range=(0.0, 1.0))[0])

        return X

    def make_visualization(self, X):
        """ Return a histogram visualization from a single example image"""

        # Convert the color
        X = self.color_transform(X)

        # Ensure the scale
        X = self.ensure_scale(X)

        # Calculate the histogram
        hist = np.histogram(X, bins=self.nbins, range=(0.0, 1.0))

        # Save the features
        X = hist[0]

        # Save the bin definitions
        bin_edges = hist[1]
        bin_centers = (bin_edges[1:] + bin_edges[0:len(bin_edges) - 1]) / 2

        # Tuple the features and centers to stand in for a visualization
        visualization = (bin_centers, X)

        return visualization, "histo"


class CannyBinPreprocessor(Preprocessor):
    """ Applies a canny edge transform, divides up the image into quadrants, and sums up the live pixels"""

    def __init__(self, color_channel="gray", nbins=64):

        # Run Preprocessor init as well
        super().__init__(color_channel=color_channel)

        # Store arguments
        self.nbins = int(nbins/2)


    def pipeline(self, X):
        """ The list of operations to apply per each image"""

        # Convert the color
        X = self.color_transform(X)

        # Equalize for consistent contrast
        X = cv2.equalizeHist(X)

        # Ensure the scale
        X = self.ensure_scale(X)

        # Apply the canny transform
        X = canny(X)

        # X Bin and histogram
        x_summary = np.mean(X, axis=1)
        x_summary = np.histogram(x_summary, bins=self.nbins)[0]

        # Y Bin and Histogram
        y_summary = np.mean(X, axis=0)
        y_summary = np.histogram(y_summary, bins=self.nbins)[0]

        # Final output
        X = np.float64(np.append(x_summary, y_summary))

        return X

    def make_visualization(self, X):
        """ Apply a canny edge transform, divide up the image into quadrants, and sum up the live pixels"""

        # Convert the color
        X = self.color_transform(X)

        # Equalize for consent contrast
        X = cv2.equalizeHist(X)

        # Ensure the scale
        X = self.ensure_scale(X)

        # Apply the canny transform
        X = canny(X)

        # X Bin and histogram
        x_summary = np.mean(X, axis=0)
        x_summary = np.histogram(x_summary, bins=self.nbins)
        x_features = x_summary[0]
        x_edges = x_summary[1]
        x_centers = (x_edges[1:] + x_edges[0:len(x_edges) - 1]) / 2

        # Y Bin and Histogram
        y_summary = np.mean(X, axis=1)
        y_summary = np.histogram(y_summary, bins=self.nbins)
        y_features = y_summary[0]
        y_edges = y_summary[1]
        y_centers = (y_edges[1:] + y_edges[0:len(y_edges) - 1]) / 2

        # Tuple the features and centers to stand in for a visualization
        visualization = [(y_centers, y_features), (x_centers, x_features)]

        return visualization, "canny"