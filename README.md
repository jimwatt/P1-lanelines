# Finding Lane Lines

A computer vision pipeline written in Python for detecting lane lines in imagery or streaming video of a roadway.  The basic pipeline uses Canny edge detection, the hough transform to detect lines, and RANSAC linear regression to estimate line location.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes using basic examples.

### Prerequisites

Imported packages include:

```
matplotlib
numpy
cv2
moviepy
sklearn
```

### Installing

No install is required --- simply clone this project from GitHub:

```
git clone https://github.com/jimwatt/lanelines.git
```

## Running the tests

The main python script is P1.py in the top directory.  This script must be run with **ipython**.


#### Help

To get usage and help,

```
ipython P1.py -- -h
```

### Process images

To process all images in the ./test_images directory,

```
ipython P1.py
```

### Process videos

To process all videos in the ./test_videos directory,

```
ipython P1.py -- --video
```

<!--## Deployment

Add additional notes about how to deploy this on a live system
-->
<!--## Built With

* [Dropwizard](http://www.dropwizard.io/1.0.2/docs/) - The web framework used
* [Maven](https://maven.apache.org/) - Dependency Management
* [ROME](https://rometools.github.io/rome/) - Used to generate RSS Feeds-->

<!--## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.-->

<!--## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). -->

## Authors

* **James Watt**

<!--## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details-->

## Acknowledgments
This project is a submission to the Udacity Self-Driving Car nanodegree:

* <https://github.com/udacity/CarND-LaneLines-P1.git>

