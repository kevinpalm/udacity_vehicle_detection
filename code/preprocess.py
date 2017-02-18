from sklearn.base import TransformerMixin
from skimage.feature import hog
import cv2
import numpy as np

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
            output = input[:,:,0]
        # Blue
        elif self.color_channel == "blue":
            output = input[:,:,0]
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

        pass


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
    """ Preprocessor for Hog features"""

    def __init__(self, color_channel="gray", pix_per_cell=8, cell_per_block=2, orient=2):

        # Run Proprocessor init as well
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

        return visualization


