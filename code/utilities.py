from os import walk
import pandas as pd
import matplotlib.pyplot as plt

def get_image_locations():
    """ Get a list of all the training images that are going to be used for training data """

    # Create a placeholder list
    locations = []

    # Iterate through the directories
    for dirpath, dirnames, filenames in walk("../input_images"):

        # Iterate through the files
        for file in filenames:
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


def stratified_subsample(df, n=1000):
    """ Retrive a subsample with an equal proportion of training classes """

    # Sample each class
    sample_v = df[df["vehicle"] == 1.0].sample(n=int(n/2))
    sample_nv = df[df["vehicle"] == 0.0].sample(n=int(n / 2))

    # Append together
    output = sample_v.append(sample_nv)

    return output


def main():

    # Retreive the file locations
    training_file_locations = get_image_locations()

    # Format the file locations to a pandas dataframe
    training_df = format_image_locations(training_file_locations)

    # Summarize the class proportions
    # summarize_classes(training_df)

    # Subsample
    subsample = stratified_subsample(training_df)

    print(subsample)


if __name__ == '__main__':
    main()