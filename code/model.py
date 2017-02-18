from preprocess import HogPreprocessor
import pandas as pd
import numpy as np


def main():

    # Load in the data
    labels = pd.read_pickle("../model_saves/training_labels.pkl")
    data = np.load("../model_saves/training_data.pkl")


if __name__ == '__main__':
    main()
