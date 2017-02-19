from preprocess import HogPreprocessor, HistoPreprocessor
import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


def main():

    # Load in the data
    labels = pd.read_pickle("../model_saves/training_labels.pkl")
    data = np.load("../model_saves/training_data.pkl")


if __name__ == '__main__':
    main()
