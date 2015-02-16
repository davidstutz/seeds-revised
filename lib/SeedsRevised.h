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
#include <string>
#include <assert.h>

#ifndef SEEDS_REVISED_H
#define	SEEDS_REVISED_H

/**
 * For OpenCV3 and OpenCV2 compatibility:
 */
#if CV_MAJOR_VERSION > 2
    #define SEEDS_REVISED_OPENCV_BGR2Lab cv::COLOR_BGR2Lab
    #define SEEDS_REVISED_OPENCV_BGR2YCrCb cv::COLOR_BGR2YCrCb
    #define SEEDS_REVISED_OPENCV_BGR2GRAY cv::COLOR_BGR2GRAY
    #define SEEDS_REVISED_OPENCV_GRAY2BGR cv::COLOR_GRAY2BGR
#else
    #define SEEDS_REVISED_OPENCV_BGR2Lab CV_BGR2Lab 
    #define SEEDS_REVISED_OPENCV_BGR2YCrCb CV_BGR2YCrCb
    #define SEEDS_REVISED_OPENCV_BGR2GRAY CV_BGR2GRAY
    #define SEEDS_REVISED_OPENCV_GRAY2BGR CV_GRAY2BGR
#endif

/**
 * Can be used when in development mode. The algorithm will throw errors whenever
 * an inconsistent state is detected.
 * 
 * However, this will slow down the algorithm!
 */
// #define DEBUG

/**
 * Set this flag when aiming to use uniform binning for color histograms.
 * 
 * The results presented in [2] are based on non-uniform binning.
 * 
 * [2] D. Stutz, A. Hermans, B. Leibe.
 *     Superpixel Segmentation using Depth Information.
 *     Bachelor thesis, RWTH Aachen University, Aachen, Germany, 2014.
 */
// #define UNIFORM

/**
 * For pixel and block updates, a memory can be used to record which pixels or blocks
 * have been changed and which pixels and blocks need to be checked again.
 * This will speed up the runtime as only few blocks and pixel updates are necessary.
 */
#define MEMORY

/**
 * If using MEMORY, this flag will add an additional heuristic speeding up the
 * algorithm but slightly decreasing the quality of the generated superpixel segmentation.
 */
// #define HEURISTIC_MEMORY

/**
 * The class SEEDS represents an implementation of SEEDS as described in [1]:
 * 
 * [1] M. van den Bergh, X. Boix, G. Roig, B. de Capitani, L. van Gool.
 *     SEEDS: Superpixels extracted via energy-driven sampling.
 *     Proceedings of the European Conference on Computer Vision, pages 13–26, 2012.
 * 
 * @author David Stutz
 */
class SEEDSRevised {

public:
    
    /**
     * Constructor, instantiates a new SEEDSRevised object with the given parameters.
     * 
     * The main parameters are:
     * 
     * * number of bins for the color histograms;
     * * number of levels;
     * * block width at first level;
     * * block height at first level.
     * 
     * The number of bins influences the quality of the resulting superpixel
     * segmentation. The number of levels and the minimum block size influence
     * the number of generated superpixels as follows. Given W x H to be the size
     * of the image, L to be the number of levels, w x h to be the minimum block size,
     * the number of superpixels is given by:
     * 
     *  floor(W / (w * 2^(L - 1))) * floor(H / (h * 2^(L - 1))).
     * 
     * Alternatively a ONE PARAMETER constructor or a ZERO PARAMETER constructor
     * may be used.
     * 
     * After instantiation, an image can be oversegmented using:
     * 
     *  seeds.initialize(image)
     *  seeds.iterate(iterations)
     * 
     * where image is an OpenCV color image and iterations is the number of
     * iterations at each level.
     * 
     * @param cv::Mat image image to be oversegmented
     * @param int numberOfLevels number of levels to use for block updates
     * @param int minimumBlockWidth block width on first level
     * @param int minimumBlockHeight block height on first level
     * @param int numberOfBins number of bins to use for the color histograms
     * @param int neighborhoodSize the (2*neighborhoodSize + 2) x (2*neighborhoodSize + 1) region around a pixel used for the smoothing prior
     * @param float minimumConfidence minimum difference in histogram intersection needed to accept a block update
     */
    SEEDSRevised(const cv::Mat& image, int numberOfLevels, int minimumBlockWidth, int minimumBlockHeight, int numberOfBins = 5, int neighborhoodSize = 1, float minimumConfidence = 0.1);

    /**
     * Constructor, instantiates a new SEEDSRevised object with the given parameters.
     * 
     * This is the ONE PARAMETER version of the constructor. The constructor will
     * derive all needed parameters given the number of superpixels. Optionally,
     * the number of bins used for the color histograms to be specified.
     * 
     * After instantiation, an image can be oversegmented using:
     * 
     *  seeds.initialize(image)
     *  seeds.iterate(iterations)
     * 
     * where image is an OpenCV color image and iterations is the number of
     * iterations at each level.
     * 
     * @param cv::Mat image image to be oversegmented
     * @param int desiredNumberOfSuperpixels desired number of superpixels
     * @param int numberOfBins number of bins for the color histograms
     * @param int neighborhoodSize the (2*neighborhoodSize + 2) x (2*neighborhoodSize + 1) region around a pixel used for the smoothing prior
     * @param float minimumConfidence minimum difference in histogram intersection needed to accept a block update
     */
    SEEDSRevised(const cv::Mat& image, int desiredNumberOfSuperpixels, int numberOfBins = 5, int neighborhoodSize = 1, float minimumConfidence = 0.1);

    /**
     * Destructor removes all initialized arrays and objects, among which are
     * the labels, the histograms and a copy of the input image.
     */
    virtual ~SEEDSRevised();

    /**
     * Get the number of superpixels which are computed according to the
     * number of levels and minimum block size used.
     * 
     * When using the alternative ONE PARAMETER constructor, the computed number
     * of superpixels may not match the desired number of superpixels perfectly
     * due to rounding issues. If requiring the number of superpixels to be matched
     * perfectly, use the original constructor.
     * 
     * @return
     */
    int getNumberOfSuperpixels() const;

    /**
     * Get the current level of the algorithm.
     * 
     * @return 
     */
    int getLevel() const;

    /**
     * Get the computed labels as two-dimensional array.
     * 
     * If used within the iteration of the algorithm (before the bottom level
     * is reached through goDownOneLevel), the array will not have the same
     * size as the image and correspond to block labels at the current level.
     * 
     * @return
     */
    int** getLabels() const;

    /**
     * Set the number of levels to use. The number of levels influences the 
     * number of superpixels to be computed. See the documentation of the
     * constructor.
     * 
     * @param int numberOfLevels
     */
    void setNumberOfLevels(int numberOfLevels);

    /**
     * Set the minimum block size. This will define the size of blocks at level 
     * one and influences the number of superpixels to be computed. See the documentation
     * of the constructor.
     * 
     * @param int minimumBlockWidth
     * @param int minimumBlockHeight
     */
    void setMinimumBlockSize(int minimumBlockWidth, int minimumBlockHeight);

    /**
     * Set the minimum confidence. The minimum confidence defines the minimum
     * difference of histogram intersection needed to accept a block update.
     * 
     * @param float minimumConfidence
     */
    void setMinimumConfidence(float minimumConfidence);

    /**
     * Set the number of bins used for the histograms.
     * 
     * If using three color channels, the actual number of bins will be the
     * number of bins to the power of three.
     * 
     * @param int numberOfBins
     */
    void setNumberOfBins(int numberOfBins);

    /**
     * Set the neighborhood size. This defines the smoothing term.
     * If N is the neighborhood size, the pixel updates will have a look
     * at a (2*N + 1) x (2*N + 2) neighborhood around the current pixel
     * and the direction to move the pixel to.
     * 
     * @param int neighborhoodSize
     */
    void setNeighborhoodSize(int neighborhoodSize);

    /**
     * Initialize the algorithm on the given image. After initialization,
     * iterations can be run using the iterate method.
     * 
     * For a different image, initialization needs to be done again.
     */
    virtual void initialize();

    /**
     * Get block width at the given level.
     * 
     * @param int level
     * @return 
     */
    int getBlockWidth(int level) const;

    /**
     * Get horizontal number of blocks at the given level.
     * 
     * @param int level
     * @return 
     */
    int getBlockWidthNumber(int level) const;

    /**
     * Get block height at the given level.
     * 
     * @param int level
     * @return 
     */
    int getBlockHeight(int level) const;

    /**
     * Get vertical number of blocks at the given level.
     * 
     * @param int level
     * @return 
     */
    int getBlockHeightNumber(int level) const;

    /**
     * Go down one level. Here, the labels are adapted to the new number
     * of blocks.
     */
    void goDownOneLevel();

    /**
     * Run iterations. The algorithm will do iterations iterations at all block
     * levels and 2*iterations iterations at the pixel level.
     * 
     * @param int iterations
     */
    virtual void iterate(int iterations);

    /**
     * Perform a block update for the given block.
     * 
     * @param int i
     * @param int j
     */
    virtual void performBlockUpdate(int i, int j);

    /**
     * Perform a pixel update for the given pixel.
     * 
     * @param int i
     * @param int j
     */
    virtual void performPixelUpdate(int i, int j);

    /**
     * Spatial memory needs to be "refreshed" before beginning iterations
     * at a new level.
     */
    virtual void reinitializeSpatialMemory();

protected:

    /**
     * Proxy for multiple constructors.
     */
    void construct(const cv::Mat &image, int numberOfBins, int numberOfLevels, int minimumBlockWidth, int minimumBlockHeight, int neighborhoodSize, float minimumConfidence);
    
    /**
     * Initialize labels. In the end, each pixel will be assigned a label. During block updates,
     * labels will be kept per block.
     */
    void initializeLabels();

    /**
     * Histograms are built level-wise beginning with the first level.
     */
    void initializeHistograms();

    /**
     * Compute the histogram intersection between the current block histogram
     * and the given superpixel histogram.
     * 
     * @param int iFrom
     * @param int jFrom
     * @param int iSuperpixelFrom
     * @param int jSuperpixelFrom
     * @return 
     */
    virtual inline float scoreCurrentBlockSegmentation(int iFrom, int jFrom, int iSuperpixelFrom, int jSuperpixelFrom) {
        float currentScore = 0.;
        float difference = 0.;

        float superpixelMinusBlockPixels = this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom] - this->pixels[this->currentLevel - 1][iFrom][jFrom];
        float blockPixels = this->pixels[this->currentLevel - 1][iFrom][jFrom];

        for (int k = 0; k < this->histogramSize; ++k) {

            if (this->histograms[this->currentLevel - 1][iFrom][jFrom][k] > 0 
                    && this->histograms[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom][k] > this->histograms[this->currentLevel - 1][iFrom][jFrom][k]) {

                difference = this->histograms[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom][k] - this->histograms[this->currentLevel - 1][iFrom][jFrom][k];
                currentScore += std::min(difference/superpixelMinusBlockPixels, this->histograms[this->currentLevel - 1][iFrom][jFrom][k]/blockPixels);
            }
        }

        return currentScore;
    }

    /**
     * Compute the intersection between the current block histogram and the new 
     * superpixel histogram.
     * 
     * @param int iFrom
     * @param int jFrom
     * @param int iSuperpixelTo
     * @param int jSuperpixelTo
     * @return 
     */
    virtual inline float scoreProposedBlockSegmentation(int iFrom, int jFrom, int iSuperpixelTo, int jSuperpixelTo) {
        float proposedScore = 0.;

        float superpixelPixels = this->pixels[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo];
        float blockPixels = this->pixels[this->currentLevel - 1][iFrom][jFrom];

        for (int k = 0; k < this->histogramSize; ++k) {

            if (this->histograms[this->currentLevel - 1][iFrom][jFrom][k] > 0
                    && this->histograms[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo][k] > 0) {

                proposedScore += std::min(this->histograms[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo][k]/superpixelPixels, this->histograms[this->currentLevel - 1][iFrom][jFrom][k]/blockPixels);
            }
        }

        return  proposedScore;
    }

    /**
     * Assign the given block to the new label, updating histogram and pixels.
     * 
     * @param int iFrom
     * @param int jFrom
     * @param int iTo
     * @param int jTo
     * @param int iSuperpixelFrom
     * @param int jSuperpixelFrom
     * @param int iSuperpixelTo
     * @param int jSuperpixelTo
     * @param int iPlusOne
     * @param int iMinusOne
     * @param int jPlusOne
     * @param int jMinusOne
     */
    virtual inline void updateBlock(int iFrom, int jFrom, int iTo, int jTo, int iSuperpixelFrom, int jSuperpixelFrom, int iSuperpixelTo, int jSuperpixelTo, int iPlusOne, int iMinusOne, int jPlusOne, int jMinusOne) {
        this->currentLabels[iFrom][jFrom] = this->currentLabels[iTo][jTo];

        this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom] -= this->pixels[this->currentLevel - 1][iFrom][jFrom];
        this->pixels[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo] += this->pixels[this->currentLevel - 1][iFrom][jFrom];

        for (int k = 0; k < this->histogramSize; ++k) {
            this->histograms[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom][k] -= this->histograms[this->currentLevel - 1][iFrom][jFrom][k];
            this->histograms[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo][k] += this->histograms[this->currentLevel - 1][iFrom][jFrom][k];
        }

        #ifdef MEMORY
            #ifdef HEURISTIC_MEMORY
                this->spatialMemory[iFrom][jFrom] = true;
                this->spatialMemory[iPlusOne][jFrom] = this->spatialMemory[iPlusOne][jFrom] || (this->currentLabels[iPlusOne][jFrom] != this->currentLabels[iFrom][jFrom]);
                this->spatialMemory[iMinusOne][jFrom] = this->spatialMemory[iMinusOne][jFrom] || (this->currentLabels[iMinusOne][jFrom] != this->currentLabels[iFrom][jFrom]);
                this->spatialMemory[iFrom][jPlusOne] = this->spatialMemory[iFrom][jPlusOne] || (this->currentLabels[iFrom][jPlusOne] != this->currentLabels[iFrom][jFrom]);
                this->spatialMemory[iFrom][jMinusOne] = this->spatialMemory[iFrom][jMinusOne] || (this->currentLabels[iFrom][jMinusOne] != this->currentLabels[iFrom][jFrom]);
            #else
                this->spatialMemory[iFrom][jFrom] = true;
                this->spatialMemory[iPlusOne][jFrom] = true;
                this->spatialMemory[iMinusOne][jFrom] = true;
                this->spatialMemory[iFrom][jPlusOne] = true;
                this->spatialMemory[iFrom][jMinusOne] = true;
            #endif
        #endif

        #ifdef DEBUG
            int sumFrom = 0;
            int sumTo = 0;
            for (int k = 0; k < this->histogramSize; ++k) {
                sumFrom += this->histograms[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom][k];
                sumTo += this->histograms[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo][k];
            }

            assert(sumFrom == this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom]);
            assert(sumTo == this->pixels[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo]);
        #endif
    }

    /**
     * Compute the probability of the current pixel belonging to the given
     * superpixel.
     * 
     * @param int iFrom
     * @param int jFrom
     * @param int iSuperpixelFrom
     * @param int jSuperpixelFrom
     * @return 
     */
    virtual inline float scoreCurrentPixelSegmentation(int iFrom, int jFrom, int iSuperpixelFrom, int jSuperpixelFrom) {
        #ifdef DEBUG
            assert(this->histograms[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom][this->histogramBins[iFrom][jFrom]] <= this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom]);
        #endif

        return ((float) this->histograms[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom][this->histogramBins[iFrom][jFrom]])/((float) this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom]);
    }

    /**
     * Compute the probability of the current pixel belonging to the
     * new superpixel.
     * 
     * @param int iFrom
     * @param int jFrom
     * @param int iSuperpixelTo
     * @param int jSuperpixelTo
     * @return 
     */
    virtual inline float scoreProposedPixelSegmentation(int iFrom, int jFrom, int iSuperpixelTo, int jSuperpixelTo) {
        #ifdef DEBUG
            assert(this->histograms[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo][this->histogramBins[iFrom][jFrom]] <= this->pixels[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo]);
        #endif

        return ((float) this->histograms[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo][this->histogramBins[iFrom][jFrom]])/((float) this->pixels[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo]);

    }

    /**
     * Add smoothing prior.
     * 
     * @param int iFrom
     * @param int jFrom
     * @param int iTo
     * @param int jTo
     * @param float currentScore
     * @param float proposedScore
     * @return 
     */
    virtual inline float scorePixelUpdate(int iFrom, int jFrom, int iTo, int jTo, float currentScore, float proposedScore) {

        if (this->neighborhoodSize > 0) {

            int labelFrom = this->currentLabels[iFrom][jFrom];
            int labelTo = this->currentLabels[iTo][jTo];

            int countFrom = 0;
            int countTo = 0;

            int iStart = std::max(0, std::min(iFrom, iTo) - this->neighborhoodSize);
            int iEnd = std::min(this->currentBlockHeightNumber, std::max(iFrom, iTo) + this->neighborhoodSize + 1);

            int jStart = std::max(0, std::min(jFrom, jTo) - this->neighborhoodSize);
            int jEnd = std::min(this->currentBlockWidthNumber, std::max(jFrom, jTo) + this->neighborhoodSize + 1);

            for (int i = iStart; i < iEnd; ++i) {
                for (int j = jStart; j < jEnd; ++j) {
                    if (this->currentLabels[i][j] == labelFrom) {
                        ++countFrom;
                    }
                    else if (this->currentLabels[i][j] == labelTo) {
                        ++countTo;
                    }
                }
            }

            currentScore *= countFrom;
            proposedScore *= countTo;
        }

        return proposedScore - currentScore;
    }

    /**
     * Assign the given pixel to the new superpixel.
     * 
     * @param int iFrom
     * @param int jFrom
     * @param int iTo
     * @param int jTo
     * @param int iSuperpixelFrom
     * @param int jSuperpixelFrom
     * @param int iSuperpixelTo
     * @param int jSuperpixelTo
     * @param int iPlusOne
     * @param int iMinusOne
     * @param int jPlusOne
     * @param int jMinusOne
     */
    virtual inline void updatePixel(int iFrom, int jFrom, int iTo, int jTo, int iSuperpixelFrom, int jSuperpixelFrom, int iSuperpixelTo, int jSuperpixelTo, int iPlusOne, int iMinusOne, int jPlusOne, int jMinusOne) {
        this->currentLabels[iFrom][jFrom] = this->currentLabels[iTo][jTo];

        --this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom];
        ++this->pixels[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo];

        --this->histograms[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom][this->histogramBins[iFrom][jFrom]];
        ++this->histograms[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo][this->histogramBins[iFrom][jFrom]];

        #ifdef MEMORY
            #ifdef HEURISTIC_MEMORY
                this->spatialMemory[iFrom][jFrom] = true;
                this->spatialMemory[iPlusOne][jFrom] = this->spatialMemory[iPlusOne][jFrom] || (this->currentLabels[iPlusOne][jFrom] != this->currentLabels[iFrom][jFrom]);
                this->spatialMemory[iMinusOne][jFrom] = this->spatialMemory[iMinusOne][jFrom] || (this->currentLabels[iMinusOne][jFrom] != this->currentLabels[iFrom][jFrom]);
                this->spatialMemory[iFrom][jPlusOne] = this->spatialMemory[iFrom][jPlusOne] || (this->currentLabels[iFrom][jPlusOne] != this->currentLabels[iFrom][jFrom]);
                this->spatialMemory[iFrom][jMinusOne] = this->spatialMemory[iFrom][jMinusOne] || (this->currentLabels[iFrom][jMinusOne] != this->currentLabels[iFrom][jFrom]);
            #else
                this->spatialMemory[iFrom][jFrom] = true;
                this->spatialMemory[iPlusOne][jFrom] = true;
                this->spatialMemory[iMinusOne][jFrom] = true;
                this->spatialMemory[iFrom][jPlusOne] = true;
                this->spatialMemory[iFrom][jMinusOne] = true;
            #endif
        #endif

        #ifdef DEBUG
            int sumFrom = 0;
            int sumTo = 0;
            for (int k = 0; k < this->histogramSize; ++k) {
                sumFrom += this->histograms[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom][k];
                sumTo += this->histograms[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo][k];
            }

            assert(sumFrom == this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom]);
            assert(sumTo == this->pixels[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo]);
        #endif
    }

    /**
     * Get the first index for the given superpixel label.
     * 
     * @param int label
     * @return 
     */
    inline int getSuperpixelIFromLabel(int label) const {
        int index = label/this->superpixelWidthNumber;

        #ifdef DEBUG
            assert(index < this->superpixelHeightNumber);
            assert(index >= 0);
        #endif

        return index;
    }

    /**
     * Get the second index for the given superpixel label
     * .
     * @param int label
     * @return 
     */
    inline int getSuperpixelJFromLabel(int label) const {
        int index = label % this->superpixelWidthNumber;

        #ifdef DEBUG
            assert(index < this->superpixelWidthNumber);
            assert(index >= 0);
        #endif

        return index;
    }
    
    /**
     * Check whether the given movement in vertical forward direction
     * would split a superpixel.
     * 
     * @param int iFrom
     * @param int jFrom
     * @param int iPlusOne
     * @param int iMinusOne
     * @param int jPlusOne
     * @param int jMinusOne
     * @return 
     */
    inline bool checkSplitVerticalForward(int iFrom, int jFrom, int iPlusOne, int iMinusOne, int jPlusOne, int jMinusOne) {

        // We consider the 3 x 3 neighborhood of the current block with label l22:
        // ------------- ^
        // |l11|l12|l13| |
        // ------------- |
        // |l21|l22|l23| | vertical
        // ------------- |
        // |l31|l32|l33| |
        // ------------- v
        // <----------->
        //   horizontal

        int l11 = this->currentLabels[iMinusOne][jMinusOne];
        int l12 = this->currentLabels[iMinusOne][jFrom];
        int l13 = this->currentLabels[iMinusOne][jPlusOne];
        int l21 = this->currentLabels[iFrom][jMinusOne];
        int l22 = this->currentLabels[iFrom][jFrom];
        int l23 = this->currentLabels[iFrom][jPlusOne];

        // To avoid border splits.
        if (iFrom == 0) {
            l11 = -1;
            l12 = -1;
            l13 = -1;
        }

        if (jFrom == 0) {
            l11 = -1;
            l21 = -1;
        }

        if (jFrom == this->width - 1) {
            l13 = -1;
            l23 = -1;
        }

        if (l12 != l22 && l21 == l22 && l23 == l22) {
            return true;
        }
        if (l11 != l22 && l12 == l22 && l21 == l22) {
            return true;
        }
        if (l13 != l22 && l12 == l22 && l23 == l22) {
            return true;
        }

        return false;
    }

    /**
     * Check whether the given movement in vertical backward direction
     * would split a superpixel.
     * 
     * @param int iFrom
     * @param int jFrom
     * @param int iPlusOne
     * @param int iMinusOne
     * @param int jPlusOne
     * @param int jMinusOne
     * @return 
     */
    inline bool checkSplitVerticalBackward(int iFrom, int jFrom, int iPlusOne, int iMinusOne, int jPlusOne, int jMinusOne) {

        // We consider the 3 x 3 neighborhood of the current block with label l22:
        // ------------- ^
        // |l11|l12|l13| |
        // ------------- |
        // |l21|l22|l23| | vertical
        // ------------- |
        // |l31|l32|l33| |
        // ------------- v
        // <----------->
        //   horizontal

        int l21 = this->currentLabels[iFrom][jMinusOne];
        int l22 = this->currentLabels[iFrom][jFrom];
        int l23 = this->currentLabels[iFrom][jPlusOne];
        int l31 = this->currentLabels[iPlusOne][jMinusOne];
        int l32 = this->currentLabels[iPlusOne][jFrom];
        int l33 = this->currentLabels[iPlusOne][jPlusOne];

        // To avoid border splits.
        if (iFrom == this->height - 1) {
            l31 = -1;
            l32 = -1;
            l33 = -1;
        }

        if (jFrom == 0) {
            l21 = -1;
            l31 = -1;
        }

        if (jFrom == this->width - 1) {
            l23 = -1;
            l33 = -1;
        }

        if (l32 != l22 && l21 == l22 && l23 == l22) {
            return true;
        }
        if (l31 != l22 && l21 == l22 && l32 == l22) {
            return true;
        }
        if (l33 != l22 && l32 == l22 && l23 == l22) {
            return true;
        }

        return false;
    }

    /**
     * Check whether the given movement in horizontal forward direction would
     * split a superpixel.
     * 
     * @param int iFrom
     * @param int jFrom
     * @param int iPlusOne
     * @param int iMinusOne
     * @param int jPlusOne
     * @param int jMinusOne
     * @return 
     */
    inline bool checkSplitHorizontalForward(int iFrom, int jFrom, int iPlusOne, int iMinusOne, int jPlusOne, int jMinusOne) {

        // We consider the 3 x 3 neighborhood of the current block with label l22:
        // ------------- ^
        // |l11|l12|l13| |
        // ------------- |
        // |l21|l22|l23| | vertical
        // ------------- |
        // |l31|l32|l33| |
        // ------------- v
        // <----------->
        //   horizontal

        int l11 = this->currentLabels[iMinusOne][jMinusOne];
        int l12 = this->currentLabels[iMinusOne][jFrom];
        int l21 = this->currentLabels[iFrom][jMinusOne];
        int l22 = this->currentLabels[iFrom][jFrom];
        int l31 = this->currentLabels[iPlusOne][jMinusOne];
        int l32 = this->currentLabels[iPlusOne][jFrom];

        // To avoid border splits.
        if (iFrom == 0) {
            l11 = -1;
            l12 = -1;
        }

        if (iFrom == this->height - 1) {
            l31 = -1;
            l32 = -1;
        }

        if (jFrom == 0) {
            l11 = -1;
            l21 = -1;
            l31 = -1;
        }

        if (l21 != l22 && l12 == l22 && l32 == l22) {
            return true;
        }
        if (l11 != l22 && l12 == l22 && l21 == l22) {
            return true;
        }
        if (l31 != l22 && l21 == l22 && l32 == l22) {
            return true;
        }

        return false;
    }

    /**
     * Check whether the given movement in horizontal backward direction would
     * split a superpixel.
     * 
     * @param int iFrom
     * @param int jFrom
     * @param int iPlusOne
     * @param int iMinusOne
     * @param int jPlusOne
     * @param int jMinusOne
     * @return 
     */
    inline bool checkSplitHorizontalBackward(int iFrom, int jFrom, int iPlusOne, int iMinusOne, int jPlusOne, int jMinusOne) {

        // We consider the 3 x 3 neighborhood of the current block with label l22:
        // ------------- ^
        // |l11|l12|l13| |
        // ------------- |
        // |l21|l22|l23| | vertical
        // ------------- |
        // |l31|l32|l33| |
        // ------------- v
        // <----------->
        //   horizontal

        int l12 = this->currentLabels[iMinusOne][jFrom];
        int l13 = this->currentLabels[iMinusOne][jPlusOne];
        int l22 = this->currentLabels[iFrom][jFrom];
        int l23 = this->currentLabels[iFrom][jPlusOne];
        int l32 = this->currentLabels[iPlusOne][jFrom];
        int l33 = this->currentLabels[iPlusOne][jPlusOne];

        // To avoid border splits.
        if (iFrom == 0) {
            l12 = -1;
            l13 = -1;
        }

        if (iFrom == this->height - 1) {
            l32 = -1;
            l33 = -1;
        }

        if (jFrom == this->width - 1) {
            l13 = -1;
            l23 = -1;
            l33 = -1;
        }

        if (l23 != l22 && l12 == l22 && l32 == l22) {
            return true;
        }
        if (l13 != l22 && l12 == l22 && l23 == l22) {
            return true;
        }
        if (l33 != l22 && l23 == l22 && l32 == l22) {
            return true;
        }
        return false;
    }

    /**
     * A copy of the image, it is converted to 8 bit channels corresponding
     * to Lab color space.
     */
    cv::Mat* image;
    /**
     * The height of the image.
     */
    int height;
    /**
     * The width of the image.
     */
    int width;
    /**
     * Boolean whether image has been initialized.
     */
    bool initializedImage;

    /**
     * The number of levels.
     */
    int numberOfLevels;
    /**
     * Width of block at level 1.
     */
    int minimumBlockWidth;
    /**
     * Height of the block at level 1.
     */
    int minimumBlockHeight;
    /**
     * Boolean defining whether a superpixel may vanish. If set to x,
     * the algorithm will ensure that each superpixel has at least x pixels.
     */
    int minimumNumberOfSublabels;
    /**
     * The size of the neighborhood region used for the smoothing term. If
     * greater than zero, for pixel updates, the algorithm will investigate
     * a (2*neightborhoodSize + 2) x (2*neighborhoodSize + 1) region around the
     * pixel and its movement direction.
     */
    int neighborhoodSize;
    /**
     * Minimum confidence used for block updates, that is the minimum required
     * difference in histogram intersection necessary for a valid block update.
     */
    float minimumConfidence;
    /**
     * The number of bins used for color histograms in one channel. This means
     * when using color images, the histograms will have a total of numberOfBins
     * to the power of three bins.
     */
    int numberOfBins;

    /**
     * The current labels: At pixel level these will be the current superpixels,
     * at a block level, these correspond to block labelings.
     */
    int** currentLabels;
    /**
     * The current level.
     */
    int currentLevel;
    /**
     * The block width at the current level.
     */
    int currentBlockWidth;
    /**
     * The block height at the current level.
     */
    int currentBlockHeight;
    /**
     * The number of blocks in horizontal direction at the current level.
     */
    int currentBlockWidthNumber;
    /**
     * The number of blocks in vertical direction at the current level.
     */
    int currentBlockHeightNumber;
    /**
     * The number of superpixels in horizontal direction.
     */
    int superpixelWidthNumber;
    /**
     * The number of superpixels in vertical direction.
     */
    int superpixelHeightNumber;
    /**
     * The width of the initial superpixels.
     */
    int superpixelWidth;
    /**
     * The height of the initial superpixels.
     */
    int superpixelHeight;
    /**
     * Boolean whether the labels are initialized.
     */
    bool initializedLabels;

    /**
     * Color histograms at all levels including superpixels.
     */
    int**** histograms;
    /**
     * Pixel counts for all blocks and superpixels.
     */
    int*** pixels;
    /**
     * The dimension of each histogram = 3 for color images.
     */
    int histogramDimensions;
    /**
     *The total size of each histogram = numberOfBins^3 for color images.
     */
    int histogramSize;
    /**
     * The histogram bin assigned to each pixel stored in a two-dimensional array.
     */
    int** histogramBins;
    /**
     * Boolean whether the histograms have been initialized.
     */
    bool initializedHistograms;

    /**
     * Memory used to speed up the algorithm.
     */
    bool** spatialMemory;
};

/**
 * The class SEEDSMeanPixels implements SEEDS with mean pixel updates. This 
 * means that pixels are exchanged between superpixels based on color difference.
 * 
 * Additionally, this implementation uses spatial pixel coordinates to
 * obtain compact and regular superpixels.
 * 
 * Mean pixel updates offer better performance but decrease speed.
 */
class SEEDSRevisedMeanPixels : public SEEDSRevised {

public:

    /**
     * Constructor, instantiates a new SEEDSRevisedMeanPixels object with the given parameters.
     * 
     * The main parameters are:
     * 
     * * number of levels;
     * * block width at first level;
     * * block height at first level.
     * 
     * The number of levels and the minimum block size influence
     * the number of generated superpixels as follows. Given W x H to be the size
     * of the image, L to be the number of levels, w x h to be the minimum block size,
     * the number of superpixels is given by:
     * 
     *  floor(W / (w * 2^(L - 1))) * floor(H / (h * 2^(L - 1))).
     * 
     * Alternatively a ONE PARAMETER constructor or a ZERO PARAMETER constructor
     * may be used.
     * 
     * After instantiation, an image can be oversegmented using:
     * 
     *  seeds.initialize(image)
     *  seeds.iterate(iterations)
     * 
     * where image is an OpenCV color image and iterations is the number of
     * iterations at each level.
     * 
     * @param cv::Mat image image to be oversegmented
     * @param int numberOfLevels number of levels to use for block updates
     * @param int minimumBlockWidth block width on first level
     * @param int minimumBlockHeight block height on first level
     * @param int numberOfBins number of bins to use for the color histograms
     * @param int neighborhoodSize the (2*neighborhoodSize + 2) x (2*neighborhoodSize + 1) region around a pixel used for the smoothing prior
     * @param float minimumConfidence minimum difference in histogram intersection needed to accept a block update
     * @param float spatialWeight weight of spatial term for compact superpixels, float between 0 and 1
     */
    SEEDSRevisedMeanPixels(const cv::Mat& image, int numberOfLevels, int minimumBlockWidth, int minimumBlockHeight, int numberOfBins = 5, int neighborhoodSize = 1, float minimumConfidence = 0.1, float spatialWeight = 0.25);

    /**
     * Constructor, instantiates a new SEEDSRevisedMeanPixels object with the given parameters.
     * 
     * This is the ONE PARAMETER version of the constructor. The constructor will
     * derive all needed parameters given the number of superpixels. Optionally,
     * the number of bins used for the color histograms to be specified.
     * 
     * After instantiation, an image can be oversegmented using:
     * 
     *  seeds.initialize(image)
     *  seeds.iterate(iterations)
     * 
     * where image is an OpenCV color image and iterations is the number of
     * iterations at each level.
     * 
     * @param cv::Mat image image to be oversegmented
     * @param int desiredNumberOfSuperpixels desired number of superpixels
     * @param int numberOfBins number of bins for the color histograms
     * @param int neighborhoodSize the (2*neighborhoodSize + 2) x (2*neighborhoodSize + 1) region around a pixel used for the smoothing prior
     * @param float minimumConfidence minimum difference in histogram intersection needed to accept a block update
     * @param float spatialWeight weight of spatial term for compact superpixels, float between 0 and 1
     */
    SEEDSRevisedMeanPixels(const cv::Mat& image, int desiredNumberOfSuperpixels, int numberOfBins = 5, int neighborhoodSize = 1, float minimumConfidence = 0.1, float spatialWeight = 0.25);
    
    /**
     * Destructor.
     */
    virtual ~SEEDSRevisedMeanPixels();

    /**
     * Set the weight for the smoothing term.
     * 
     * @param float spatialWeight
     */
    void setSpatialWeight(float spatialWeight);
    
    /**
     * Run iterations. The algorithm will do iterations iterations at all block
     * levels and 2*iterations iterations at the pixel level.
     * 
     * @param int iterations
     */
    virtual void iterate(int iterations);

protected:

    /**
     * Before pixel updates, the means need to be initialized.
     */
    virtual void initializeMeans();

    /**
     * Assign the given pixel to the new superpixel.
     * 
     * @param int iFrom
     * @param int jFrom
     * @param int iTo
     * @param int jTo
     * @param int iSuperpixelFrom
     * @param int jSuperpixelFrom
     * @param int iSuperpixelTo
     * @param int jSuperpixelTo
     * @param int iPlusOne
     * @param int iMinusOne
     * @param int jPlusOne
     * @param int jMinusOne
     */
    virtual inline void updatePixel(int iFrom, int jFrom, int iTo, int jTo, int iSuperpixelFrom, int jSuperpixelFrom, int iSuperpixelTo, int jSuperpixelTo, int iPlusOne, int iMinusOne, int jPlusOne, int jMinusOne) {

        SEEDSRevised::updatePixel(iFrom, jFrom, iTo, jTo, iSuperpixelFrom, jSuperpixelFrom, iSuperpixelTo, jSuperpixelTo, iPlusOne, iMinusOne, jPlusOne, jMinusOne);

        for (int k = 0; k < this->meanDimensions; ++k) {
            this->means[1][iSuperpixelFrom][jSuperpixelFrom][k] -= this->means[0][iFrom][jFrom][k];
            this->means[1][iSuperpixelTo][jSuperpixelTo][k] += this->means[0][iFrom][jFrom][k];
        }

        #ifdef DEBUG
            float mean = 0.;
            for (int k = 0; k < this->histogramDimensions; ++k) {
                mean = this->means[1][iSuperpixelFrom][jSuperpixelFrom][k]/this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom];
                assert(mean <= 255);

                mean = this->means[1][iSuperpixelTo][jSuperpixelTo][k]/this->pixels[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo];
                assert(mean <= 255);
            }
        #endif
    }

    /**
     * Compute the probability of the current pixel belonging to the given
     * superpixel.
     * 
     * @param int iFrom
     * @param int jFrom
     * @param int iSuperpixelFrom
     * @param int jSuperpixelFrom
     * @return 
     */
    virtual inline float scoreCurrentPixelSegmentation(int iFrom, int jFrom, int iSuperpixelFrom, int jSuperpixelFrom) {
        float currentColorScore = 0.;

        if (this->histogramDimensions == 1) {
            float difference = this->means[1][iSuperpixelFrom][jSuperpixelFrom][0]/this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom] - this->means[0][iFrom][jFrom][0];

            currentColorScore = difference*difference/this->colorNormalization;
        }
        else {
            float differenceL = this->means[1][iSuperpixelFrom][jSuperpixelFrom][0]/this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom] - this->means[0][iFrom][jFrom][0];
            float differenceA = this->means[1][iSuperpixelFrom][jSuperpixelFrom][1]/this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom] - this->means[0][iFrom][jFrom][1];
            float differenceB = this->means[1][iSuperpixelFrom][jSuperpixelFrom][2]/this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom] - this->means[0][iFrom][jFrom][2];

            currentColorScore = (differenceL*differenceL + differenceA*differenceA + differenceB*differenceB)/this->colorNormalization;
        }

        #ifdef DEBUG
            assert(currentColorScore <= 1 && currentColorScore >= 0);
        #endif

        if (this->spatialWeight > 0) {
            float differenceX = this->means[1][iSuperpixelFrom][jSuperpixelFrom][this->meanDimensions - 2]/this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom] - this->means[0][iFrom][jFrom][this->meanDimensions - 2];
            float differenceY = this->means[1][iSuperpixelFrom][jSuperpixelFrom][this->meanDimensions - 1]/this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom] - this->means[0][iFrom][jFrom][this->meanDimensions - 1];
            float currentSpatialScore = (differenceX*differenceX + differenceY*differenceY)/this->spatialNormalization;

            #ifdef DEBUG
                assert(currentSpatialScore <= 1 && currentSpatialScore >= 0);
            #endif

            return (1 - this->spatialWeight)*currentColorScore + this->spatialWeight*currentSpatialScore;
        }
            
        return currentColorScore;
    }

    /**
     * Compute the probability of the current pixel belonging to the
     * new superpixel.
     * 
     * @param int iFrom
     * @param int jFrom
     * @param int iSuperpixelTo
     * @param int jSuperpixelTo
     * @return 
     */
    virtual inline float scoreProposedPixelSegmentation(int iFrom, int jFrom, int iSuperpixelTo, int jSuperpixelTo) {
        float proposedColorScore = 0.;

        if (this->histogramDimensions == 1) {
            float difference = this->means[1][iSuperpixelTo][jSuperpixelTo][0]/this->pixels[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo] - this->means[0][iFrom][jFrom][0];

            proposedColorScore = difference*difference/this->colorNormalization;
        }
        else {
            float differenceL = this->means[1][iSuperpixelTo][jSuperpixelTo][0]/this->pixels[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo] - this->means[0][iFrom][jFrom][0];
            float differenceA = this->means[1][iSuperpixelTo][jSuperpixelTo][1]/this->pixels[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo] - this->means[0][iFrom][jFrom][1];
            float differenceB = this->means[1][iSuperpixelTo][jSuperpixelTo][2]/this->pixels[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo] - this->means[0][iFrom][jFrom][2];

            proposedColorScore = (differenceL*differenceL + differenceA*differenceA + differenceB*differenceB)/this->colorNormalization;
        }

        #ifdef DEBUG
            assert(proposedColorScore <= 1 && proposedColorScore >= 0);
        #endif

        if (this->spatialWeight > 0) {
            float differenceX = this->means[1][iSuperpixelTo][jSuperpixelTo][this->meanDimensions - 2]/this->pixels[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo] - this->means[0][iFrom][jFrom][this->meanDimensions - 2];
            float differenceY = this->means[1][iSuperpixelTo][jSuperpixelTo][this->meanDimensions - 1]/this->pixels[this->numberOfLevels - 1][iSuperpixelTo][jSuperpixelTo] - this->means[0][iFrom][jFrom][this->meanDimensions - 1];
            float proposedSpatialScore = (differenceX*differenceX + differenceY*differenceY)/this->spatialNormalization;

            #ifdef DEBUG
                assert(proposedSpatialScore <= 1 && proposedSpatialScore >= 0);
            #endif

            return (1 - this->spatialWeight)*proposedColorScore + this->spatialWeight*proposedSpatialScore;
        }
            
        return proposedColorScore;
    }

    /**
     * Add smoothing prior.
     * 
     * @param int iFrom
     * @param int jFrom
     * @param int iTo
     * @param int jTo
     * @param float currentScore
     * @param float proposedScore
     * @return 
     */
    virtual inline float scorePixelUpdate(int iFrom, int jFrom, int iTo, int jTo, float currentScore, float proposedScore) {

        if (this->neighborhoodSize > 0) {

            int countFrom = 0;
            int countTo = 0;

            int labelFrom = this->currentLabels[iFrom][jFrom];
            int labelTo = this->currentLabels[iTo][jTo];

            int iStart = std::max(0, std::min(iFrom, iTo) - this->neighborhoodSize);
            int iEnd = std::min(this->currentBlockHeightNumber, std::max(iFrom, iTo) + this->neighborhoodSize + 1);

            int jStart = std::max(0, std::min(jFrom, jTo) - this->neighborhoodSize);
            int jEnd = std::min(this->currentBlockWidthNumber, std::max(jFrom, jTo) + this->neighborhoodSize + 1);

            for (int i = iStart; i < iEnd; ++i) {
                for (int j = jStart; j < jEnd; ++j) {
                    if (this->currentLabels[i][j] == labelFrom) {
                        ++countFrom;
                    }
                    else if (this->currentLabels[i][j] == labelTo) {
                        ++countTo;
                    }
                }
            }

            currentScore /= countFrom;
            proposedScore /= countTo;
        }

        return currentScore - proposedScore;
    }

    int meanDimensions;
    float**** means;
    bool initializedMeans;
    float colorNormalization;
    float spatialWeight;
    float spatialNormalization;

};

#endif	/* SEEDS_REVISED_H */
