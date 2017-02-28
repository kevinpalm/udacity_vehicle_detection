**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/class_frequencies.jpg
[image2]: ./output_images/gray_hog.jpg
[image3]: ./output_images/red_histogram.jpg
[image4]: ./output_images/canny_bin_histogram.jpg
[image5]: ./output_images/sample_training_estimates.jpg
[image6]: ./output_images/search_windows.jpg
[image7]: ./output_images/active_windows.jpg
[image8]: ./output_images/heatmap.jpg
[image9]: ./output_images/final_classify.jpg
[video1]: ./processed_project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I ended up creating a sklearn style transformer class to handle extracting my HOG features, for convenience of being able to use the transform in a sklearn pipeline. The code for this class is contained in lines 100 through 147 of the file called `preprocess.py`.

I started by reading in all the `vehicle` and `non-vehicle` images.  I plotted a summary of the frequency of each label, to make sure my training data was roughly balanced:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`). I had pretty good success using the lightness channel from HLS, but ended up using a histogram equalized gray color channel.

Here is an example using the `gray` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:

![alt text][image2]

In additional to HOG features, I used color histograms of the red, green, and blue channels as features.

![alt text][image3]

Finally, because I was having so much trouble with false positives, I engineered one additional feature which I called 'canny bins'. For this feature, I converted to a monochromatic channel (my final estimator uses gray), histogram equalized, applied a canny transform, then summarized the number of active pixels per x value and the number of active pixels per y value. Finally, the x and y counts were summarized as x and y histograms. My intuition on creating this feature was the cars should be generally more geometric and have more occurences of long straight canny edges, and the using the feature definitely improved my testing scores.

![alt text][image4]

####2. Explain how you settled on your final choice of HOG parameters.

This was actually sort of my downfall. At the very begining of this project, I started off using only HOG features and a linear SVM. I explored all the RGB, Grayscale, and HLS color channels and tinkered with HOG parameters, just to get a feel for the predictive power of HOG features alone. I ended up deciding to keep things basic, use grayscale and my default HOG parameters, which I thought were `orientations=9`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`.

Turns out I made an error as I jumped into the more fancy stuff of exploring different classifiers and playing with alternative features. I'd accidently set my default as `orientations=2`. To top it off, I also didn't start out with any form of histogram equalization before taking HOG features. And so a jumped into a lot of cool tests with my hands tied behind my back.

I think I just generally underestimated how important the HOG features are to this project, and so I didn't look to my preprocessing steps when the troubleshooting started. I also didn't have a well designed accuracy test at that point - I was randomly splitting my training data, when I should have been separating the GTI data - and so it just wasn't clicking why I had so many false positives in my video frames.

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

My preprocessors and estimator is defined in lines 56-75 of `model.py`, and is trained in lines 91-126. Each preprocessor/estimator is stored to a file as a sklearn pipeline pickle file for use making predictions.

I'm currently using a sklearn's VotingClassifier with three sklearn estimators - Perceptron(), RidgeClassifier(alpha=1000), and LinearSVC(). In a local test using GTI examples as the testing examples and everything else as training examples, the accuracy and F1 scores are around 0.85.

Here's a few training examples fed through the pipeline:

![alt text][image5]

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I used three scales of windows: 64x64, 96x96, and 128x128. All of them have 0.5 overlap, and focus on the bottom half of the image, but the smaller scales extend less far towards the bottom. The code defining the windows is from lines 183 through 199 in `model.py`, and the code for formating the windows is in 6-41 in `boxes.py`.

I decided on the scales and overlap by experimentation - I found that using three scales led to a reasonable range of near/far cars, and 0.5 was about right for not getting completely overrun by false positives in concentrated regions.

![alt text][image6]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on three scales using grayscale HOG features, histograms of color, and histograms of canny edge bins in the feature vector, which works okay.

![alt text][image7]

After running through all the windows, the image gets represented as a heatmap of active windows and overlap.

![alt text][image8]

Then a threshold gets applied to clean out some of the false positives before drawing the final bounding boxes.

![alt text][image9]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./processed_project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap, and combined it with a decayed copy of all the previous heatmaps (for each past frame, the old heatmap is multiplied by 0.8 before adding in the current frame). Then a threshold is applied to help with false positives.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.

Lines 219 through 262 in `model.py` contain my final classifier that is used in the Moviepy VideoFileClip().fl_image() function.

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

I had major problems getting my false-positives under control. While my pipeline still isn't working that well, it is a lot better than it was.

My top issues so far that I've overcome are:
* Getting a good train-test split for accuracy and f1 tests
* Not looking closely enough at my HOG features, or knowing how much I needed to depend on them
* Not using any histogram equalization prior to creating HOG features or color histograms

Regardless of my improvements, my final video result is pretty janky still. I feel like there's probably some relatively easy gains available if I had more time for tinkering with final estimators, frame-by-frame combinations, and heatmap thresholds. At least, though, I think I'm relatively happy with my input features and preprocessors finally.

It's pretty easy to see where my pipeline will fail. Extreme shadowing, irregular road marks, trees... it doesn't take much to trip it up and trigger false positives. As well, very dark black cars seem to escape classification depending on the angle.

At this point, when thinking about how to make it more robust, (I'm tempted to just say use a deep neural network, hah) I think it just needs more tuning, but I'm a little afraid that there's another little error somewhere along the pipeline.

Also, the pipeline is nowhere near running real-time. I understand the implimentation for image-wide feature extraction for each frame, but I feel like I'm not settled enough on my approach to warrent implimenting it yet.

