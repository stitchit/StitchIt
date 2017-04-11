# StitchIt
Akanksha Periwal - aperiwal

Sai Harshini Nimmala - snimmala

## Optimization and Parallelization of Image Stitching

We plan to optimize the image stitching pipeline that is popularly available as an OpenCV library through application of explicit parallelism directives such as those offered by Halide on a CPU. Parts of the algorithm will also be ported to a GPU to analyze and ideally improve the time required to perform the image stitching.



### Background

Image stitching is the process of combining one or more overlapping images into the end result of one seamless panoramic picture. OpenCV offers a popularly used library that performs image stitching. Its implemenation however lacks any significant optimizations and is found to be inherently slow. The image processing pipeline involved in this algorithm has scope for parallelization which we wish to exploit.
The several stages in the image processing pipeline to perform the stitch are:

  1. Keypoint Detection and Extraction: 
    Extract salient points in an image that have distinctive features

  2. Computing Feature Descriptors (SIFT/SURF/BRIEF): 
    Characterize the extracted keypoints with a feature descriptor that helps describe the region around them and find correspondences between sets of images
    
  3. Feature Matching: 
    Calculating the similarity between the obtained feature descriptors and computing matching pairs using nearest neighbours found
  
  4. Calculate Homography (using RANSAC): 
    Find a 3x3 transformation matrix to compute how a planar scene would look from a second camera location given only the first camera image
    
  5. Warping, Blending & Composition: 
    Transform the images into a single frame and seamlessly blend them at the edges where they overlap



### The Challenge

  * Homography calculation using RANSAC requires a large amount of iterative computation making it difficult to parallelize. It could become a bottleneck for the entire algorithm. There have been very few attempts previously to parallelize this computation.
  
  * In the case of high-resolution images, memory bandwidth could lead to a bottleneck.
  
  * Approaching the problem poses many parallelization constructs to choose from. Selecting the right combination of primitives for the different stages of the pipeline can be tricky and overwhelming.
  


### Resources & Platform Choice

  * We plan to implement the algorithm on a single-CPU-multi-core platform as well as a GPU. The required resources are available through the GHC machine clusters.
  
  * We will use the source code for the image stitching class from the openCV library as the reference code for our project. [Check it out!](https://github.com/opencv/opencv/tree/master/modules/stitching/src). This code is written in C++, which works well with a lot of parallelization APIs and libraries. Python being an interpreted language would be much slower, making C++ a much better choice since we are concerned about performance more than ease of coding.
  
  * We plan to explore Halide to help us apply our image processing optimizations more rapidly. Further, Halide is embedded into C++, making our language choice effective.



### Goals and Deliverables

  * To significantly improve execution time of image stitching compared to OpenCV's stitcher class
  
  * Determine execution time differences between different stages of our implementation (on CPU/on GPU/combinations thereof) compared to the baseline OpenCV code
  
  * Study the impact of the stitcher on images with different resolutions
  
  * Determine the tradeoffs between computation speed and communication overhead when using a GPU implementation for specific stages of the pipeline
  


### Schedule

  * Week 1
    - Study the base code on OpenCV and determine the opportunities for parallelism
    - Get acquainted with Halide
    - Parallelize stage 1: keypoint detection and extraction

  * Week 2
    - Choose a suitable descriptor
    - Parallelize stages 2, 3 and 5: Computing feature descriptors, feature matching, warping, and blending on a CPU

  * Week 3
    - Attempt to parallelize stage 4: homography computation
    - Implement select stages on GPU and analyze performance

  * Week 4
    - Determine scheduling of the different strategies to obtain max. performance
    - Figure out bottlenecks and come up with workarounds
    - Test with multi-resolution images

  * Stretch Goal
    - Apply stitcher algorithm to multiple images and test scalability
    
