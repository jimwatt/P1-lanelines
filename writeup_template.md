# **Finding Lane Lines on the Road** 
####James Watt  
Udacity Self-Driving Car Nanodegree:
Project 1

---

## 1. Goal

Make a pipeline that finds and annotates lane lines in video of a roadway.

[//]: # (Image References)

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

## 2. The Computational Pipeline.

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
Use the Canny algorithm to detect images in the smoothed grayscale image.
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
Use RANSAC linear regression to generate a single representative left lane line, and single representative right lane line. 
![alt text][image8]


## 3. Results

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
<video width="520" height="320" controls src="test_videos_output/SolidWhiteRight.mp4" frameborder="0" allowfullscreen></video>

##### Solid Yellow Left
<video width="520" height="320" controls src="test_videos_output/SolidYellowLeft.mp4" frameborder="0" allowfullscreen></video>

##### Challenge
<video width="520" height="320" controls src="test_videos_output/challenge.mp4" frameborder="0" allowfullscreen></video>

## 4. Code Repository
My code implementation for this pipeline is in the following GitHub repository.

## 5. Potential Shortcomings




## 6. Possible Improvements


