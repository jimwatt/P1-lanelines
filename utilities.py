# coding: utf-8
#####################################################################
# Imports
#####################################################################
import cv2
from sklearn import linear_model        # for ransac linear regression
import numpy as np

######################################################################
# Generic helper functions
######################################################################

# Convert an image to grayscale
def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

# Apply canny edge detection
def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

# Apply a Gaussian kernel to smooth an image
def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

# Mask an image by the polygon defined in vertices
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)   
    ignore_mask_color = 255  
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image, mask

# Draw lines on an image.
def draw_lines(img, lines, color=[255, 0, 0], thickness=2):
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img, (x1, y1), (x2, y2), color, thickness)

# Appply the hough transform to detect lines in an image
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# Cobine two images with weightings.
def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    return cv2.addWeighted(initial_img, α, img, β, λ)


##############################################################################
# Sub-Algorithms for detecting lane lines
##############################################################################

# Given a set of lines, determine which belong to the left and right lane lines.
def separateLeftRightLines(lines,imx,imy):
    # Given the image size imx,imy, sort lines into left and right using slope and location
    left = []
    right = []
    da = 0.1
    for line in lines:
        x1,y1,x2,y2 = line[0]        
        angle = np.arctan2(y2-y1,x2-x1)
        if(angle<0):        #make all angles between 0 an pi.
            angle += np.pi
        if(  angle>(da*np.pi) and angle<(0.5-da)*np.pi and 0.5*(x1+x2)>imx/2+20):
            right.append(line)
        elif(angle>(0.5+da)*np.pi and angle<(1.0-da)*np.pi  and 0.5*(x1+x2)<imx/2-20):
            left.append(line)
    return left,right

# Sample points along a given line
def sampleLine(x1,y1,x2,y2,ds): 
    # Given a line defined by endpoints (x1,y1) ad (x2,y2), return points sampled along the line with spacing ds   
    p1,p2 = np.array([x1,y1]),np.array([x2,y2])
    length = np.linalg.norm(p2-p1);     # distance between endpoints
    numpts = np.int(length/ds)+1
    ptsx = np.linspace(x1,x2,numpts)    
    ptsy = np.linspace(y1,y2,numpts)  
    return np.vstack((ptsx,ptsy))

# Use RANSAC linear regression to average multiple lines.
def averageLines(lines,xvals):
    # Given lines, compute a single representative line using ransac linear regression to reject outliers, and evaluate that line for
    # the given xvals. 
    # First, sample points from lines
    allpts = np.empty((2,0), np.float)
    for line in lines:
        pts = sampleLine(*line[0],1.0)
        allpts = np.append(allpts,pts,axis=1)
    # Then perform the regression, evaluate for xvals, and return the corresponding yvals 
    # lr = linear_model.LinearRegression()
    lr = linear_model.RANSACRegressor()
    lr.fit(allpts[0,:].reshape(-1, 1), allpts[1,:].reshape(-1, 1))
    return lr.predict(xvals.reshape(-1, 1))

# Get the vertices for the region of interest polytope.
def getVertices(image,vertextype):
    # The challenge.mp4 movie requires a different vertex set to define the mask region.
    imshape = image.shape
    imx = imshape[1]
    imy = imshape[0]

    if vertextype=='standard':
        dx = 30
        dy = 65
        dxtop = 60
        return np.array([[(dx,imy),(imx/2-dxtop, imy/2+dy), (imx/2+dxtop, imy/2+dy), (imx-dx,imy)]], dtype=np.int32)
    elif vertextype=='challenge':
        dx = 180
        dy = 100
        dxtop = 90
        dybot = 60
        return np.array([[(dx,imy-dybot),(imx/2-dxtop, imy/2+dy), (imx/2+dxtop, imy/2+dy), (imx-dx,imy-dybot)]], dtype=np.int32)
    else:
        print('ERROR: unknown vertex type {} in getVertices()'.format(vertextype))
        assert(False)



