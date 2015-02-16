/**
 * Implementation of the superpixel algorithm called SEEDS [1] described in
 * 
 *  [1] M. van den Bergh, X. Boix, G. Roig, B. de Capitani, L. van Gool.
 *      SEEDS: Superpixels extracted via energy-driven sampling.
 *      Proceedings of the European Conference on Computer Vision, pages 13–26, 2012.
 * 
 * If you use this code, please cite [1] and [2]:
 * 
 *  [2] D. Stutz, A. Hermans, B. Leibe.
 *      Superpixel Segmentation using Depth Information.
 *      Bachelor thesis, RWTH Aachen University, Aachen, Germany, 2014.
 * 
 * [2] is available online at 
 * 
 *      http://davidstutz.de/bachelor-thesis-superpixel-segmentation-using-depth-information/
 * 
 * Note that all results published in [2] are based on an extended version
 * of the Berkeley Segmentation Benchmark [3], the Berkeley Segmentation
 * Dataset [3] and the NYU Depth Dataset [4].
 * 
 * [3] P. Arbeláez, M. Maire, C. Fowlkes, J. Malik.
 *     Contour detection and hierarchical image segmentation.
 *     Transactions on Pattern Analysis and Machine Intelligence, 33(5):898–916, 2011.
 * [4] N. Silberman, D. Hoiem, P. Kohli, R. Fergus.
 *     Indoor segmentation and support inference from RGBD images.
 *     Proceedings of the European Conference on Computer Vision, pages 746–760, 2012.
 * 
 * The extended version of the Berkeley Segmentation Benchmark will be
 * published on GitHub [6]:
 * 
 *  [5] https://github.com/davidstutz
 * 
 * The code is published under the BSD 3-Clause:
 * 
 * Copyright (c) 2014 - 2015, David Stutz
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 * 
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. Neither the name of the copyright holder nor the names of its contributors
 *    may be used to endorse or promote products derived from this software
 *    without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
 * THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */
#include <opencv2/opencv.hpp>
#include <boost/timer.hpp>
#include <boost/filesystem.hpp>
#include <boost/filesystem/fstream.hpp>

#ifndef SEEDS_REVISED_TOOLS_H
#define	SEEDS_REVISED_TOOLS_H

        
/**
 * Class Integrity provides some helper to check the integrity of the generated
 * superpixel segmentations.
 * 
 * @author David Stutz
 */
class Integrity {

public:

    /**
     * Computes the actually number of superpixels generated.
     * 
     * @param SEEDSRevised seeds
     * @param int rows
     * @param int cols
     * @return
     */
    static int countSuperpixels(int** labels, int rows, int cols);

    /**
     * Given the labels, relabels them in place.
     * 
     * @param int** labels superpixel labels  (first dimension is x-axis)
     * @param int rows
     * @param int cols
     */
    static void relabel(int** labels, int rows, int cols);
};

/**
 * Class Export provides a helper to export a matrix to CSV format.
 * 
 * @author David Stutz
 */
class Export {

public:

    /**
     * Save labels to CSV file.
     * 
     * @param int** labels superpixel labels  (first dimension is x-axis)
     * @param int
     * @param int
     * @param boost::filesystem::path path path to store CSV file
     */
    static void CSV(int** labels, int rows, int cols, boost::filesystem::path path);
    
    /**
     * Save the given OpenCV matrix in BSD evaluation file format, as for example:
     * 
     *  fprintf(fid, '%10d %10g\n', ...)
     * 
     * @param matrix
     * @param precision
     * @param path
     */
    template <typename T>
    static void BSDEvaluationFile(const cv::Mat &matrix, int precision, boost::filesystem::path path);
};

/**
 * Class Draw provides helpers to visualize the generated superpixel segmentations.
 * 
 * @author David Stutz
 */
class Draw {

public:

    /**
     * Draws contours around superpixels.
     * 
     * Code adapted from code provided by the authors of [6]:
     * 
     * [6] R. Achanta, A. Shaji, K. Smith, A. Lucchi, P. Fua, S. Süsstrunk.
     *     SLIC superpixels.
     *     Technical report, École Polytechnique Fédérale de Lausanne, 2010.
     * 
     * @param int** labels superpixel labels (first dimension is x-axis)
     * @param cv::Mat image original image
     * @param int* rgb rgb color of contours
     * @return 
     */
    static cv::Mat contourImage(int** labels, const cv::Mat &image, int* bgr);

    /**
     * Draws a colored label image where each label gets assigned a 
     * random color.
     * 
     * @param int** labels superpixel labels (first dimension is x-axis)
     * @param cv::Mat image original image
     * @return
     */
    static cv::Mat labelImage(int** labels, const cv::Mat &image);
    
    /**
     * Compute a mean image, that is every superpixel is colored 
     * according to its mean color.
     * 
     * @param int** labels superpixel labels (first dimension is x-axis)
     * @param image original image
     * @return 
     */
    static cv::Mat meanImage(int** labels, const cv::Mat &image);

};

#endif	/* SEEDS_REVISED_TOOLS_H */

