
# coding: utf-8

##############################################################################################################
# import packages
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import math
import os
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import utilities as ut

###################################################################################################################
# The main process_image() function takes an image as input, and returns the same image with lane lines annotated.
def process_image(image,vertextype):

    # Define parameters for the various transforms and algrotihms:
    kernel_size = 5         # size of Gaussian smoothing kernel
    low_threshold = 50      # low-threshold for canny edge detection
    high_threshold = 150    # high-threshold for canny edge detection
    rho = 1                 # distance resolution in pixels of the Hough grid
    theta = np.pi/180       # angular resolution in radians of the Hough grid
    threshold = 5           # minimum number of votes (intersections in Hough grid cell)
    min_line_length = 10    # minimum number of pixels making up a line
    max_line_gap = 20       # maximum gap in pixels between connectable line segments

    # The Main Computational Pipeline:

    # 0. Get dimensions for our image
    imshape = image.shape
    imx = imshape[1]
    imy = imshape[0]

    # 1. Grayscale the image
    gray = ut.grayscale(image)
        
    # 2. Apply Gaussian smoothing
    blur_gray = ut.gaussian_blur(gray, kernel_size)

    # 3. Apply Canny edge detection
    edges = ut.canny(blur_gray, low_threshold, high_threshold)

    # 4. Apply region of interest mask
    vertices = ut.getVertices(image,vertextype)    # get the vertices for this vertextype
    masked_edges, mask = ut.region_of_interest(edges,vertices)
    
    # 5. Apply Hough Transform
    lines = ut.hough_lines(masked_edges, rho, theta, threshold, min_line_length, max_line_gap)

    # 6. Use slope and position to separate left and right lines
    leftlines,rightlines = ut.separateLeftRightLines(lines,imx,imy)

    # 7. Average the left and right lines, and annotate the image
    final_image = np.copy(image)
    if(len(leftlines)):
        leftx = np.array([0,imx/2-50],dtype=np.int)
        lefty = ut.averageLines(leftlines,leftx).astype(np.int)
        cv2.line(final_image, (leftx[0],lefty[0]), (leftx[1], lefty[1]), [255, 0, 0], 6)
    if(len(rightlines)):
        rightx = np.array([imx/2+50,imx],dtype=np.int)
        righty = ut.averageLines(rightlines,rightx).astype(np.int)
        cv2.line(final_image, (rightx[0],righty[0]), (rightx[1], righty[1]), [0, 255, 0], 6)
            
    # We are DONE!  The rest is just plotting of the steps in the pipeline for generating the write-up.

    plot_pipeline = False

    if plot_pipeline:
        print("Plotting pipeline ...")

        # Plot the original image
        plt.figure(1)
        plt.imshow(image)
        plt.title("Original Image")
        plt.savefig("pipeline/fig0_original.png")

        # Plot the grayscale image
        plt.figure(2)
        plt.imshow(gray,cmap='gray')
        plt.title("Grayscale Image")
        plt.savefig("pipeline/fig1_grayscale.png")

        # Plot the blurred image
        plt.figure(3)
        plt.imshow(blur_gray,cmap='gray')
        plt.title("Blurred Image")
        plt.savefig('pipeline/fig2_blurred.jpg')

        # Plot the canny edges
        plt.figure(4)
        plt.imshow(edges,cmap='gray')
        plt.title("Canny Edges Image")
        plt.savefig('pipeline/fig3_canny_edges.jpg')

        # Plot the region mask
        region = np.zeros_like(image)
        pts = vertices.reshape((-1,1,2))
        cv2.fillConvexPoly(region,pts,color=(255, 0, 0))
        color_mask = cv2.addWeighted(image,1.0,region,0.3,0)
        plt.figure(5)
        plt.imshow(color_mask)
        plt.title("Region Mask")
        plt.savefig('pipeline/fig4_region_mask.jpg')

        # Plot the masked canny edges
        plt.figure(6)
        plt.imshow(masked_edges,cmap='gray')
        plt.title("Masked Canny Edges Image")
        plt.savefig('pipeline/fig5_masked_edges.jpg')

        # Plot the Hough lines 
        plt.figure(7)
        line_image = np.zeros_like(image)
        ut.draw_lines(line_image, lines)
        plt.imshow(line_image)
        plt.title("Hough Lines")
        plt.savefig('pipeline/fig6_hough_lines.jpg')

        # Plot the hough lines after filtering for left and right lines
        plt.figure(8)
        lrline_image = np.zeros_like(image)
        ut.draw_lines(lrline_image, leftlines, color=[255,0,0])
        ut.draw_lines(lrline_image, rightlines, color=[0,255,0])
        plt.imshow(lrline_image)
        plt.title("Left-Right Hough Lines")
        plt.savefig('pipeline/fig7_leftright_lines.jpg')

        # Plot the averaged lines as the final image
        plt.figure(9)
        plt.imshow(final_image)
        plt.title("Lane Lines")
        plt.savefig('pipeline/fig8_lane_lines.jpg')

    return final_image

##########################################################################################
import argparse

if __name__ == '__main__':

    # Give usage, help, and parse command line arguments
    parser = argparse.ArgumentParser(description="Script for detecting and annotating lane lines in images or videos.",usage='ipython %(prog)s -- [options]')
    parser.add_argument('-v','--video', action='store_true',help="process videos instead (images are processed by default).")
    args = parser.parse_args()

    # Do we want to process the test images, the test videos, or both?
    if(args.video):
        processimages = False 
        processvideos = True
    else :
        processimages = True
        processvideos = False

    # Process images
    if(processimages):
        print('Processing images in directory: test_images')
        imagenames = os.listdir("test_images/")
        vertextype = 'standard'
        for imagename in imagenames:    # process each image in the directory
            fullname = os.path.join('test_images', imagename)
            image = mpimg.imread(fullname)
            procimg = process_image(image,vertextype)
            
            # Save the output figures
            savename = os.path.join('test_images_output', imagename)
            plt.figure()
            plt.imshow(procimg)
            plt.title(imagename)
            plt.savefig(savename)
    
    # Process videos  
    if(processvideos):
        print('Processing videos in directory: test_videos')
        videos = os.listdir("test_videos/")
        for video in videos:            # process each video in the directory
            vertextype = 'standard'
            if video=='challenge.mp4':      # the challenge video requires a different mask region
                vertextype = 'challenge'
            processed_video = os.path.join('test_videos_output', video)
            videoclip = os.path.join('test_videos', video)
            clip1 = VideoFileClip(videoclip)
            processed_clip = clip1.fl_image(lambda image: process_image(image,vertextype)) # run the lane lines processor
            get_ipython().run_line_magic('time', 'processed_clip.write_videofile(processed_video, audio=False)')    # save the output

    print("DONE!!!")




