# SEEDS Revised

Implementation of the superpixel algorithm called SEEDS [1] described in

    [1] M. van den Bergh, X. Boix, G. Roig, B. de Capitani, L. van Gool.
        SEEDS: Superpixels extracted via energy-driven sampling.
        Proceedings of the European Conference on Computer Vision, pages 13–26, 2012.

If you use this code, please cite [1] and [2]:

    [2] D. Stutz, A. Hermans, B. Leibe.
        Superpixel Segmentation using Depth Information.
        Bachelor thesis, RWTH Aachen University, Aachen, Germany, 2014.

**Note:** Evaluation results are now available online (to view or download) at [http://davidstutz.de/projects/superpixelsseeds/](http://davidstutz.de/projects/superpixelsseeds/).

[2] is available online at [http://davidstutz.de/bachelor-thesis-superpixel-segmentation-using-depth-information/](http://davidstutz.de/bachelor-thesis-superpixel-segmentation-using-depth-information/).

Note that all results published in [2] are based on an extended version of the Berkeley Segmentation Benchmark [3], the Berkeley Segmentation Dataset [3] and the NYU Depth Dataset [4].

    [3] P. Arbeláez, M. Maire, C. Fowlkes, J. Malik.
        Contour detection and hierarchical image segmentation.
        Transactions on Pattern Analysis and Machine Intelligence, 33(5):898–916, 2011.
    [4] N. Silberman, D. Hoiem, P. Kohli, R. Fergus.
        Indoor segmentation and support inference from RGBD images.
        Proceedings of the European Conference on Computer Vision, pages 746–760, 2012.

The extended version of the Berkeley Segmentation Benchmark is available on GitHub: [https://github.com/davidstutz/extended-berkeley-segmentation-benchmark](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark).

![Example: several superpixel segmentations.](screenshot.png?raw=true "Example: several superpixel segmentations")

## Compile

SEEDS Revised is easily compiled using [CMake](http://www.cmake.org/):

    # Clone the repository:
    $ git clone https://github.com/davidstutz/seeds-revised.git
    # Go to the build subfolder to generate the CMake files:
    $ cd seeds-revised/build
    $ cmake ..
    # Compile the library and corresponding Command Line Interface:
    $ make

The binaries will be saved to `seeds-revised/bin`. The command line interface offers the following options:

    $ ../bin/cli --help
    Allowed options:
        --help                          produce help message
        --input arg                     the folder to process, may contain several 
                                  images
        --bins arg (=5)                 number of bins used for color histograms
        --neighborhood arg (=1)         neighborhood size used for smoothing prior
        --confidence arg (=0.100000001) minimum confidence used for block update
        --iterations arg (=2)           iterations at each level
        --spatial-weight arg (=0.25)    spatial weight
        --superpixels arg (=400)        desired number of supüerpixels
        --verbose                       show additional information while processing
        --csv                           save segmentation as CSV file
        --contour                       save contour image of segmentation
        --labels                        save label image of segmentation
        --mean                          save mean colored image of segmentation
        --output arg (=output)          specify the output directory (default is 
                                  ./output)

## Usage

The library contains two classes:

* `SEEDSRevised`: the original algorithm as proposed in [1].
* `SEEDSRevisedMeanPixels`: an extension using mean pixel updates as discussed in [1].

Thorough documentation can be found within the code. The following example will demonstrate the basic usage of `SEEDSRevisedMeanPixels`:

    #include <opencv2/opencv.hpp>
    #include "SeedsRevised.h"
    #include "Tools.h"

    // ...

    cv::Mat image = cv::imread(filepath);
    
    // Number of desired superpixels.
    int superpixels = 400;
    
    // Number of bins for color histograms (per channel).
    int numberOfBins = 5;
    
    // Size of neighborhood used for smoothing term, see [1] or [2].
    // 1 will be sufficient, >1 will slow down the algorithm.
    int neighborhoodSize = 1;
    
    // Minimum confidence, that is minimum difference of histogram intersection
    // needed for block updates: 0.1 is the default value.
    float minimumConfidene = 0.1;
    
    // The weighting of spatial smoothing for mean pixel updates - the euclidean
    // distance between pixel coordinates and mean superpixel coordinates is used
    // and weighted according to:
    //  (1 - spatialWeight)*colorDifference + spatialWeight*spatialDifference
    // The higher spatialWeight, the more compact superpixels are generated.
    float spatialWeight = 0.25;
    
    // Instantiate a new object for the given image.
    SEEDSRevisedMeanPixels seeds(image, superpixels, numberOfBins, neighborhoodSize, minimumConfidence, spatialWeight);

    // Initializes histograms and labels.
    seeds.initialize();
    // Runs a given number of block updates and pixel updates.
    seeds.iterate(iterations);
    
    // Save a contour image to the following location:
    std::string storeContours = "./contours.png";

    // bgr color for contours:
    int bgr[] = {0, 0, 204};
    
    // seeds.getLabels() returns a two-dimensional array containing the computed
    // superpixel labels.
    cv::Mat contourImage = Draw::contourImage(seeds.getLabels(), image, bgr);
    cv::imwrite(store, contourImage);

## OpenCV 3 Compatibility

The implementation is compatible with OpenCV 2 and OpenCV 3 and tries to detect the used version automatically. However, as some constants changed in OpenCV3, the code may be slightly adapted when using development releases of OpenCV3. In particular, this relates to the following constants:

    CV_BGR2GRAY
    CV_BGR2Lab

## License

Copyright (c) 2014, David Stutz
All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.