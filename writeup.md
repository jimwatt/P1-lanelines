# **Finding Lane Lines on the Road** 
####James Watt
**Udacity Self-Driving Car Nanodegree:
Project 1**

---

## 1. Goal

Make a pipeline that finds and annotates lane lines in imagery or streaming video of a roadway.

[//]: # "Image References"

[image0]: ./pipeline/fig0_original.png "Original"
[image1]: ./pipeline/fig1_grayscale.png "Grayscale"
[image2]: ./pipeline/fig2_blurred.jpg "Smoothed"
[image3]: ./pipeline/fig3_canny_edges.jpg "Canny Edges"
[image4]: ./pipeline/fig4_region_mask.jpg "Region Mask"
[image5]: ./pipeline/fig5_masked_edges.jpg "Masked Edges"
[image6]: ./pipeline/fig6_hough_lines.jpg "Hough Lines"
[image7]: ./pipeline/fig7_leftright_lines.jpg "Left-Right Lines"
[image8]: ./pipeline/fig8_lane_lines.jpg "Lane Lines"
[image9]: ./test_images_output/solidWhiteCurve.jpg "Solid White Curve"
[image10]: ./test_images_output/solidWhiteRight.jpg "Solid White Right"
[image11]: ./test_images_output/solidYellowCurve.jpg "Solid Yellow Curve"
[image12]: ./test_images_output/solidYellowCurve2.jpg "Solid Yellow Curve 2"
[image13]: ./test_images_output/solidYellowLeft.jpg "Solid Yellow Left"
[image14]: ./test_images_output/whiteCarLaneSwitch.jpg "White Car Lane Switch"

---

## 2. Code Repository
My code implementation for this pipeline is hosted in the GitHub repository:

<https://github.com/jimwatt/lanelines.git>

**Please note that I have not provided a Jupyter Notebook.**  I find development to be much faster and more cleanly organized outside of Jupyter.  My hope is that this document contains all the step by step information and results that would be provided for reviewers in a Jupyter Notebook.  

The code is chiefly contained in two scripts:

* [P1.py](./P1.py) is the main entry point for the code.
* [utilities.py](./utilities.py) provides the main algorithmic functions.

---

## 2. Running the Code

Please see the [README.md](./README.md) file for usage instructions.  Very briefly,


To get usage and help:

```
ipython P1.py -- -h
```

To process all images in the ./test_images directory:

```
ipython P1.py
```

To process all videos in the ./test_videos directory:

```
ipython P1.py -- --video
```

___

## 3. The Computational Pipeline.

Given an image (or video stream) of the roadway,
![alt text][image0]

my pipeline for annotating lane lines operates using the following seven steps: 

##### Step 1. 
Convert the original image to grayscale.
![alt text][image1]

##### Step 2. 
Smooth the image with a Gaussian kernel.
![alt text][image2]

##### Step 3.
Use the Canny algorithm to detect edges in the smoothed grayscale image.
![alt text][image3]

##### Step 4. 
Apply a region mask to select only pixels within the region of interest.
![alt text][image4]
![alt text][image5]

##### Step 5.
Apply a Hough transform to extract lines from the masked edge images.
![alt text][image6]

##### Step 6. 
Use slope and location to differentiate between left lane lines, and right lane lines.
![alt text][image7]

##### Step 7. 
Use RANSAC linear regression to generate a single representative left lane line, and single representative right lane line. The RANSAC linear regressor is implemented using scikit-learn.
![alt text][image8]
___

## 4. Results

### Processing of still images:

##### Solid White Curve
![alt text][image9]

#####Solid White Right
![alt text][image10]

#####Solid Yellow Curve
![alt text][image11]

#####Solid Yellow Curve 2
![alt text][image12]

#####Solid Yellow Left
![alt text][image13]

##### White Care Lane Switch
![alt text][image14]

### Processing of streaming video: 

##### Solid White Right
<video width="520" height="320" controls src="test_videos_output/solidWhiteRight.mp4" frameborder="0" allowfullscreen></video>

##### Solid Yellow Left
<video width="520" height="320" controls src="test_videos_output/solidYellowLeft.mp4" frameborder="0" allowfullscreen></video>

##### Challenge
<video width="520" height="320" controls src="test_videos_output/challenge.mp4" frameborder="0" allowfullscreen></video>

---
## 5. Potential Shortcomings
* The algorithms have not been tested on very much data -- only three movies.  Current performance is good on solidWhiteRight.mp4 and solidYellowLeft.mp4.  Performance is marginal on challenge.mp4.  Performance can be improved further on challenge.mp4 by further specific tuning, although we run the danger of tuning to only this movie making it fragile to new situations.  I perfer to stop further tuning at this point until a larger dataset can be obtained, ensuring more robustness and generalization. 
* I used the RANSAC linear regression algorithm for "averaging" multiple hough lines to obtain a single representative lane line.  This has the benefit of recognizing and discarding outliers (stray lines due to cat's eyes and lane markers close to the lane line, for example).  However, this approach does have a potential shortcoming.  For the case of dotted lines, if only two lines are detected and due to road curvature the two lines are at different angles, the RANSAC approach will likely entirely discard one of the lines as an outlier, rather than attempting to average the two lines.  This does not appear to be a problem for the given videos, but could be a problem in other cases.

---
## 6. Possible Improvements
* For the challenge.mpf video, it is clear that the algorithm has a difficult time when the road surface changes from asphalt to concrete, and from concrete to asphalt.  The sharp line between the two surfaces confuses the linear regression algorithm.  An improvement would be to further gate which line angles will be considered for lane lines to remove these spurious lines. We could gain better performance by considering that the lane lines do not change their relative position and relative angles very much to filter spurious lines.
* I would be interested in applying a sequential filtering approach.  At present, the algorithm processes each image independently, and no information from prior images is used to improve accuracy.  We could leverage the fact that (in global space) the lane lines do not move, and therefore we expect persistence from frame to frame.  We could use a Kalman filtering appproach to estimate parameters for the lane line.  In this approach, the images are treated as measurements to _update_ our estimates of the lane lines, rather than generating them from whole cloth at each frame.

