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
