# FINAL REPORT
                                              
# Optimization and Parallelization of Image Stitching

Image stitching is the process of combining one or more overlapping images into the end result of one seamless panoramic picture. Our aim was to analyse and try to improve the performance of different stages of the image stiching pipeline. We obtained a 1.5x speedup in the total execution time of the stiching algorithm using SIFT descriptors. We also explored the usage of optimized rotational invariant BRIEF descriptors for the feature descriptor computation, which were seen to be an order of magnitude faster than the optimized SIFT descriptors.

# Background

The image stitching pipeline can be broken up into 2 major chunks :

(a) Keypoint Detection and Feature Descriptors

![Alt text](Pipeline_FeatureMatching.jpg?raw=true "Feature")

(b) Homography and Blending

![Alt text](Pipeline_HomographyAndBlending.JPG?raw=true "Blending")

The baseline code for our implementation was OpenPano, an open-source image stitching code in C++ (coded by a CMU student who has taken 15418 in the past, therefore the code was well-written and already had all basic optimizations in place). 

----

## PART 1 - Keypoint Detection and Feature Descriptors

(A) GAUSSIAN BLURRING :

The blurring kernel is called multiple times for the DoG computation, and consumes a significant amount of execution time. Therefore, different implementations of the same were tried. All of mentioned implementations involved doing separable 1D convolutions instead of a 2-D convolution-

   * Naive CPU implementation :
This is just the straightforward way with an imaged sized temporary array following a horizontal blur, which is then convolved with a vertical blurring kernel. 
This is clearly memory-bottlenecked, and was implemented just to see the improvement other implementations give.

  * OpenPano Implementation :
The OpenPano implementation takes one row or column at a time, copies it into a temporary array and then applies the convolution on it before storing it into the result array itself.
The memory storage overhead in this case is a just an array of max(height/width) of the input image. However, the number of memory loads and stores would still be high, because the entire result array is filled by the horizontal blur before moving onto the vertical blur.

  * Using SSE2 vector intrinsics with chunking :
Here, the non-edge case pixels were convolved horizontally using SIMD instructions. However, SSE doesn't support gather instructions, so this could be done only for one of the 1D convolutions.
Storing the intermediate temp array in column major order was tried (which would again allow a horizontal blur like load instruction). However, this was not helpful in improving the execution time. 

  * Naive GPU implementation :
This was just a CUDA version of the naive CPU version.

  * GPU Implementation : Using Tiling + Shared Memory
Here, every block was responsible for a small patch of the input image.Every thread in the block would load a different pixel, and then compute the horizontal and vertical convolution on a pixel, with the intermediate temporary result array stored in the shared memory.

The two GPU implementations had execution times higher than the original implementation, so the code was profiled to see the time distribution between the CPU-GPU transfers and the actual kernel computation. In all the cases, nearly 75-80% of the time was involved in the memcpy (slightly bigger images were marginally better), thus making these GPU implementations unsuitable for images of the given size (around 400kB).

However, the feature detection involves computation multiple blurs of different scales on the same image. Thus, instead of copying the image to the device memory everytime, the same image is reused for all the different scale computations. 

The timing comparisons between the different implementations can be seen below :

![Alt text](BlurringComparisons_1 (1).jpg?raw=true "BlurringComparisons")

The SSE vector instrincs implementation is not as impressive as one would expect it to be. This is because the SIMD execution implemented in the kernel is limited to only one of the two 1-D convolutions involved, and there were additional divergent edge cases.

The GPU implementation with the intelligent memory transfers performs pretty well, and is close to, and mostly better than the CPU vector instrinsics version.

In the GPU implementations , it can be seen that the execution times improved as the number of calls to the kernel increased. Since there is no carrying of state between calls (in the first two implementations atleast), and the calls are not pipelines (which could have lead to some initial memory transfer latency hiding), this observation is rather puzzling. 
      
(B) As seen from the GPU implementation of the blurring, the communication overhead is too much between the CPU and GPU. Thus, we chose to go with a SIMD calcution of the difference of Gaussian layers, which gave the expected 3.5-4x speedup in the DoG computation.  

(C) FEATURE DESCIPTORS 

There were 2 parallel approaches taken to the Feature Descriptors -

   * The SIFT computation in the OpenPano source code was fairly well-written. A few parallelizations (in the magnitude, orientation calculations) and minor optimizations/SIMD vectorizations (in the histogram calculation) were made, leading to a small improvement in the time required for computation for the descriptors. However, the optimization possible with SIFT is fairly limited (beyond multicore parallelizations).

   * Thus, a parallel approach taken to explore optimizing the feature descriptor calculation using BRIEF descriptors.
BRIEF descriptors are binary descriptors, which are less compute intensive to compute and also to match. 

*SIFT descriptors are calculated by taking a weighted histogram of the orientations of the points in a patch around the keypoint, which need to be then matches using a Euclidean distance computation.
BRIEF descriptors on the other hand are binary strings of intensity comparisons between sampled points in a patch around the keypoint, and are matched using a simple hamming distance computation. (xor and sum of the two strings)*

The BRIEF descriptors were coded into the feature calculation algorithm, with vectorised version of Hamming distance calculation (using fast hardware popcount instructions). (Took care of reducing memory accesses and sampling point storage patterns that would lead to less cache misses)

   * However, the problem is that BRIEF descriptors are not rotationally invariant.  Adding rotational and scale invariance to them was challenging, but necessary to make it a reasonable alternative for SIFT. One way of doing that would be rotating the sub-patch in the image based on its orientation and then computing the descriptor around it. However, this is computationally very expensive.

To reduce the computational cost of adding rotational invariance, a set of sampling patterns for discrete angles of rotation were pre-computed. A simple spatially weighted mean was used to determine the orientation of a keypoint, and this orientation was used to choose the set of sampling points needed for the BRIEF descriptor computation.

The figure belows shows the matches between an image and a rotated + scaled down version of it (therefore making the BRIEF descriptor scale and rotation invariant):
     
 ![Alt text](Flower_Scale_Rotated_Matches.jpg?raw=true "BRIEF Descriptor Matching")
 
 While the descriptors give good matches now, they still not as stable as the SIFT descriptors, and don't give panoramas for some image sets.

### Results

![Alt text](SiftOptimizationGraph.jpg?raw=true "SiftOpt")

![Alt text](SIFTvsBRIEF.jpg?raw=true "SIFT vs BRIEF")

Using BRIEF descriptors over SIFT gives 10x improvement in descriptor calculation, and 3.5-4x improvement in keypoint matching(which contribute about 15% to the total execution time).

## PART 2 - Homography and Blending


### Division of Work

  * Akanksha Periwal (aperiwal) :
    Gaussian Blurring Implementations, SIFT Optimization, BRIEF descriptor implementation, rotational invariance and optimization 
    
  * Sai Harshini (snimmala) :
    Homography and Blending

---
---
   
# CHECKPOINT UPDATE

### Checkpoint Update

The initial plan was to work with the OpenCV implementation of the Stitcher Class as the baseline. However, on going through the code, we observed that that various stages of image stitching – feature detection, feature matching, homography estimation and blending – were not so distinctly seen in the code, and thus optimizing it would require a much deeper understanding of the code. A different baseline was thus needed and we decided to to go ahead with OpenPano.
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
  * Optimize the gaussian blurring kernel – work on CPU
  * Optimize extrema detection
  * Complete parallel implementation of maximum spanning tree

2. Before 5th April: 
  * Work on reducing the time required for finding the two nearest neighbours in the feature matches, and work around the problem of finding very less matches between some images (which affects the homography matrix estimation, inturn increasing the number of iterations required for RANSAC).
  * Optimize various image processing calculations such as resizing, concatenation, cropping, transformation and color computations.

3. Before 10th April:
  * Explore possible parallelization of RANSAC


   
                               PROJECT PROPOSAL

## Optimization and Parallelization of Image Stitching

We plan to optimize the image stitching pipeline that is used to create panoramas of several images through application of explicit parallelism directives such as those offered by Halide on a CPU.

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

  * We plan to implement the algorithm on a single-CPU-multi-core platform. The required resources are available through the GHC machine clusters.
  
  * We will use the source code for the image stitching class from the openCV library as the reference code for our project. [Check it out!](https://github.com/opencv/opencv/tree/master/modules/stitching/src). This code is written in C++, which works well with a lot of parallelization APIs and libraries. Python being an interpreted language would be much slower, making C++ a much better choice since we are concerned about performance more than ease of coding.
  
  * We plan to explore Halide to help us apply our image processing optimizations more rapidly. Further, Halide is embedded into C++, making our language choice effective.



### Goals and Deliverables

  * To significantly improve execution time of image stitching compared to OpenPano
  
  * Determine execution time differences between different stages of our implementation compared to the baseline code
  
  * Study the impact of the stitcher on images with different resolutions
  
