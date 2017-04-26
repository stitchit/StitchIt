## Optimization and Parallelization of Image Stitching

We plan to optimize the image stitching pipeline that is used to create panoramas of several images through application of explicit parallelism directives such as those offered by Halide on a CPU. Parts of the algorithm will also be ported to a GPU to analyze and ideally improve the time required to perform the image stitching.



### Background

Image stitching is the process of combining one or more overlapping images into the end result of one seamless panoramic picture. There are several open source image stitching suites that are available for use. The image processing pipeline involved in this algorithm has scope for parallelization which we wish to exploit.
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
  
  * We will use the source code from OpenPano, the open source panorama stitcher suite as the reference code for our project. [Check it out!](https://github.com/ppwwyyxx/OpenPano) This code is written in C++, which works well with a lot of parallelization APIs and libraries. Python being an interpreted language would be much slower, making C++ a much better choice since we are concerned about performance more than ease of coding.
  
  * We plan to explore Halide to help us apply our image processing optimizations more rapidly. Further, Halide is embedded into C++, making our language choice effective.



### Goals and Deliverables

  * To significantly improve execution time of image stitching compared to OpenPano
  
  * Determine execution time differences between different stages of our implementation (on CPU/on GPU/combinations thereof) compared to the baseline code
  
  * Study the impact of the stitcher on images with different resolutions
  
  * Determine the tradeoffs between computation speed and communication overhead when using a GPU implementation for specific stages of the pipeline
  


### Checkpoint Update

The initial plan was to work with the OpenCV implementation of the Stitcher Class as the baseline. However, on going through the code, we observed that that various stages of image stitching – feature detection, feature matching, homography estimation and blending – were not so distinctly seen in the code, and thus optimizing it would require a much deeper understanding of the code. A different baseline was thus neeeded and we decided to to go ahead with OpenPano.
Although the Stitcher class of OpenCV was stated to be very slow, several recent optimizations have contributed to considerable speedup. The OpenPano implementation also has several optimizations incorporated, making it a faster algorithm than what was expected. This has made the problem of reducing execution time more challenging for us, requiring us to come up with more innovative and indirect ways of parallelizing the code.

WORK DONE:

Understanding the code and understanding the possible opportunities for further optimization.

The image stitching can be broadly divided into 2 parts :

  * Keypoint detection, Feature Description and Matching
    * As mentioned above, the base code uses SIFT feature descriptors, which though more stable are also computationally intensive. There was a relatively naive implementation of BRIEF feature descriptors which wasn't completely incorporated into the code. However, looking at the simplicity in computing and comparing BRIEF desciptors (which uses just Hamming distance), we decided to explore using them instead of SIFT.
    * A uniform distribution was chosen for the pattern generation, and the Hamming distance code was vectorized and sped up.
    * DoG calculation was also vectorized.
    * With these changes, the feature calculation time reduced by more than half when compared to SIFT descriptors.
    * However, one key problem being faced now is the number of matches being found in case of these desciptors are much lesser compared to the previous case, making the feature matching time higher than in case of SIFT descriptors.
    * The gaussian blurring, and extrema detection are areas that can still be optimized further.

  * Transform estimation, blending and stitching
    * An area for potential optimization was found in the camera estimation mode when cameras with different focal lengths are used. After building a graph of pairwise matches of the images with matching confidence as weights, a maximum  spanning tree is built to determine the best matches. Currently, a sequential algorithm is used to compute the tree. Thus, parallelization is being attempted to increase speedup.
    
PLAN AHEAD:
1. Before 30th April:
  * Optimize the gaussian blurring kernel – work on CPU as well as GPU implementation.
  * Optimize extrema detection
  * Complete parallel implementation of maximum spanning tree

2. Before 5th April: 
  * Work on reducing the time required for finding the two nearest neighbours in the feature matches, and work around the problem of finding very less matches between some images (which affects the homography matrix estimation, inturn increasing the number of iterations required for RANSAC).
  * Optimize various image processing calculations such as resizing, concatenation, cropping, transformation and color computations.

3. Before 10th April:
  * Explore GPU implementation for the other parts of the feature descriptor calculation and compare execution times.
  * Explore possible parallelization of RANSAC


  
