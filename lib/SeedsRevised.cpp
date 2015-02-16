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
#include "SeedsRevised.h"
#include <opencv2/opencv.hpp>
#include <math.h>
#include <string>

SEEDSRevised::SEEDSRevised(const cv::Mat &image, int numberOfLevels, int minimumBlockWidth, int minimumBlockHeight, int numberOfBins, int neighborhoodSize, float minimumConfidence) {
    this->construct(image, numberOfBins, numberOfLevels, minimumBlockWidth, minimumBlockHeight, neighborhoodSize, minimumConfidence);
}

SEEDSRevised::SEEDSRevised(const cv::Mat &image, int desiredNumberOfSuperpixels, int numberOfBins, int neighborhoodSize, float minimumConfidence) {
    
    int width = image.cols;
    int height = image.rows;
    
    int minimumBlockWidths[3] = {2, 3, 4};
    int minimumBlockHeights[3] = {2, 3, 4};
    int maxLevels = 12;
    
    int minDifference = -1;
    int minLevels = 0;
    int minBlockWidth = 0;
    int minBlockHeight = 0;
    
    for (int w = 0; w < 3; ++w) {
        for (int h = 0; h < 3; ++h) {
            if (abs(minimumBlockWidths[w] - minimumBlockHeights[h]) > 1) {
                continue;
            }
            
            for (int l = 2; l < maxLevels + 1; ++l) {
                int superpixels = std::floor(width/(minimumBlockWidths[w]*pow(2, l - 1))) * std::floor(height/(minimumBlockHeights[h]*pow(2, l - 1)));
                int difference = abs(desiredNumberOfSuperpixels - superpixels);
                if (difference < minDifference || minDifference < 0) {
                    minDifference = difference;
                    minLevels = l;
                    minBlockWidth = minimumBlockWidths[w];
                    minBlockHeight = minimumBlockHeights[h];
                }
            }
        }
    }
    
    assert(minDifference >= 0);
    assert(minLevels >= 2);
    assert(minBlockWidth > 0);
    assert(minBlockHeight > 0);
    
    this->construct(image, numberOfBins, minLevels, minBlockWidth, minBlockHeight, neighborhoodSize, minimumConfidence);
}

void SEEDSRevised::construct(const cv::Mat &image, int numberOfBins, int numberOfLevels, int minimumBlockWidth, int minimumBlockHeight, int neighborhoodSize, float minimumConfidence) {
    this->numberOfLevels = numberOfLevels;
    this->minimumBlockWidth = minimumBlockWidth;
    this->minimumBlockHeight = minimumBlockHeight;
    this->numberOfBins = numberOfBins;
    this->minimumConfidence = minimumConfidence;
    this->neighborhoodSize = neighborhoodSize;
    
    this->initializedImage = false;
    this->initializedLabels = false;
    this->initializedHistograms = false;
    this->currentLevel = 0;
    this->currentBlockWidth = 0;
    this->currentBlockHeight = 0;
    this->currentBlockWidthNumber = 0;
    this->currentBlockHeightNumber = 0;
    this->superpixelWidthNumber = 0;
    this->superpixelHeightNumber = 0;
    this->superpixelWidth = 0;
    this->superpixelHeight = 0;
    this->minimumNumberOfSublabels = 1;
    this->histogramDimensions = 0;
    this->histogramSize = 0;
    
    this->image = new cv::Mat();
    int channels = image.channels();
    
    assert(channels == 1 || channels == 3);
    
    if (channels == 1) {
        image.convertTo(*this->image, CV_8UC1);
    }
    else if (channels == 3) {
        image.convertTo(*this->image, CV_8UC3);
        cv::cvtColor(*this->image, *this->image, SEEDS_REVISED_OPENCV_BGR2Lab, 3);
    }
    
    this->height = this->image->rows;
    this->width = this->image->cols;
}

SEEDSRevised::~SEEDSRevised() {
    
    if (this->initializedImage == true) {
        delete this->image;
    }
    
    if (this->initializedLabels == true) {
        
        for (int i = 0; i < this->height; ++i) {
            delete[] this->currentLabels[i];
            delete[] this->spatialMemory[i];
        }
        
        delete[] this->currentLabels;
        delete[] this->spatialMemory;
        this->initializedLabels = false;
    }
    
    if (this->initializedHistograms == true) {
        
        int blockHeightNumber;
        int blockWidthNumber;
        
        for (int level = 1; level <= this->numberOfLevels; ++level) {
            blockHeightNumber = this->getBlockHeightNumber(level);
            blockWidthNumber = this->getBlockWidthNumber(level);

            // Note that the array is built up as a matrix: the first index references
            // the row rather than the column.
            for (int i = 0; i < blockHeightNumber; ++i) {
                for (int j = 0; j < blockWidthNumber; ++j) {
                    delete[] this->histograms[level - 1][i][j];
                }
                
                delete[] this->histograms[level - 1][i];
                delete[] this->pixels[level - 1][i];
            }
            
            delete[] this->histograms[level - 1];
            delete[] this->pixels[level - 1];
        }
        
        delete[] this->histograms;
        delete[] this->pixels;
        
        for (int i = 0; i < this->height; ++i) {
            delete[] this->histogramBins[i];
        }
        
        delete[] this->histogramBins;
        this->initializedHistograms = false;
    }
}

void SEEDSRevised::setNumberOfLevels(int numberOfLevels) {
    assert(numberOfLevels >= 2);
    
    this->numberOfLevels = numberOfLevels;
}

void SEEDSRevised::setMinimumBlockSize(int minimumBlockWidth, int minimumBlockHeight) {
    assert(minimumBlockWidth > 0 && minimumBlockHeight > 0);
    assert(minimumBlockWidth*2 <= this->width && minimumBlockHeight*2 <= this->height);
    
    this->minimumBlockWidth = minimumBlockWidth;
    this->minimumBlockHeight = minimumBlockHeight;
}

void SEEDSRevised::setMinimumConfidence(float minimumConfidence) {
    assert(minimumConfidence >= 0);
    
    this->minimumConfidence = minimumConfidence;
}

void SEEDSRevised::setNeighborhoodSize(int neighborhoodSize) {
    assert(neighborhoodSize >= 0);
    
    this->neighborhoodSize = neighborhoodSize;
}

void SEEDSRevised::setNumberOfBins(int numberOfBins) {
    this->numberOfBins = numberOfBins;
}

void SEEDSRevised::initialize() {
    this->initializeLabels();
    this->initializeHistograms();
}

void SEEDSRevised::initializeLabels() {
    // The highest level is the superpixel level, level 0 is the pixel level.
    // The useNumberOfLevels method assures the numberOfLevels to be 2 or higher.
    this->currentLevel = this->numberOfLevels;
    
    // Calculate the current block width and height from minimumBlockWidth
    // and minimumBlockHeight.
    this->currentBlockWidth = this->getBlockWidth(this->currentLevel);
    this->currentBlockHeight = this->getBlockHeight(this->currentLevel);
    
    this->currentBlockWidthNumber = this->getBlockWidthNumber(this->currentLevel);
    this->currentBlockHeightNumber = this->getBlockHeightNumber(this->currentLevel);
    
    // Needs to be updated whenever the number of levels or the minimum block size is changed.
    this->superpixelWidthNumber = this->getBlockWidthNumber(this->numberOfLevels);
    this->superpixelHeightNumber = this->getBlockHeightNumber(this->numberOfLevels);
    this->superpixelWidth = this->getBlockWidth(this->numberOfLevels);
    this->superpixelHeight = this->getBlockHeight(this->numberOfLevels);
    
    int label = 0;
    
    // In the end each pixel will have a label, in the meantime we will simply only 
    // use a part of the matrix for the block labels such that we do not need
    // to resize the matrix at each level.
    this->currentLabels = new int*[this->height];
    
    // Initialize labels in blocks of 4 blocks, as 4 blocks built one superpixel
    // at the level above.
    for (int i = 0; i < this->height; ++i) {
        this->currentLabels[i] = new int[this->width];
        
        for (int j = 0; j < this->width; ++j) {
            
            if (i < this->superpixelHeightNumber && j < this->superpixelWidthNumber) {
                this->currentLabels[i][j] = label;
                ++label;
            }
            // To initialize the array properly.
            else {
                this->currentLabels[i][j] = -1;
            }
        }
    }
    
    // Spatial memory will remember which blocks or pixels have been updated in the
    // previous iteration, and for which blocks or pixels there will not be a change.
    this->spatialMemory = new bool*[this->height];
    for (int i = 0; i < this->height; ++i) {
        this->spatialMemory[i] = new bool[this->width];
        
        for (int j = 0; j < this->width; ++j) {
            this->spatialMemory[i][j] = true;
        }
    }
    
    this->goDownOneLevel();
    this->initializedLabels = true;
}

int SEEDSRevised::getBlockWidth(int level) const {
    #ifdef DEBUG
        assert(level > 0 && level <= this->numberOfLevels);
    #endif
        
    return this->minimumBlockWidth*((int) pow((double) 2, (double) (level - 1)));
}

int SEEDSRevised::getBlockWidthNumber(int level) const {
    #ifdef DEBUG
        assert(level > 0 && level <= this->numberOfLevels);
    #endif
        
    return this->width/this->getBlockWidth(level);
}

int SEEDSRevised::getBlockHeight(int level) const {
    #ifdef DEBUG
        assert(level > 0 && level <= this->numberOfLevels);
    #endif
    
    return this->minimumBlockHeight*((int) pow((double) 2, (double) (level - 1)));
}

int SEEDSRevised::getBlockHeightNumber(int level) const {
    #ifdef DEBUG
        assert(level > 0 && level <= this->numberOfLevels);
    #endif

    return this->height/this->getBlockHeight(level);
}

void SEEDSRevised::goDownOneLevel() {
    #ifdef DEBUG
        assert(this->currentLevel > 0);
    #endif
    
    --this->currentLevel;
    
    if (this->currentLevel > 0) {
        
        int newBlockWidthNumber = this->getBlockWidthNumber(this->currentLevel);
        int newBlockHeightNumber = this->getBlockHeightNumber(this->currentLevel);
        
        for (int i = this->currentBlockHeightNumber - 1; i > -1; --i) {
            for (int j = this->currentBlockWidthNumber - 1; j > -1; --j) {

                this->currentLabels[2*i][2*j] = this->currentLabels[i][j];
                this->currentLabels[2*i + 1][2*j] = this->currentLabels[i][j];
                this->currentLabels[2*i][2*j + 1] = this->currentLabels[i][j];
                this->currentLabels[2*i + 1][2*j + 1] = this->currentLabels[i][j];

                // Remember to add new diagonal pixels for the block in bottom right corner.
                if (i == this->currentBlockHeightNumber - 1 && j == this->currentBlockWidthNumber - 1) {
                    for (int k = 2*i + 2; k < newBlockHeightNumber; ++k) {
                        for (int l = 2*j + 2; l < newBlockWidthNumber; ++l) {
                            this->currentLabels[k][l] = this->currentLabels[i][j];
                            this->currentLabels[k][l] = this->currentLabels[i][j];
                        }
                    }
                }

                if (i == this->currentBlockHeightNumber - 1) {
                    for (int k = 2*i + 2; k < newBlockHeightNumber; ++k) {
                        this->currentLabels[k][2*j] = this->currentLabels[i][j];
                        this->currentLabels[k][2*j + 1] = this->currentLabels[i][j];
                    }
                }

                if (j == this->currentBlockWidthNumber - 1) {
                    for (int l = 2*j + 2; l < newBlockWidthNumber; ++l) {
                        this->currentLabels[2*i][l] = this->currentLabels[i][j];
                        this->currentLabels[2*i + 1][l] = this->currentLabels[i][j];
                    }
                }
            }
        }
        
        this->currentBlockWidthNumber = newBlockWidthNumber;
        this->currentBlockHeightNumber = newBlockHeightNumber;
        
        this->currentBlockWidth = this->getBlockWidth(this->currentLevel);
        this->currentBlockHeight = this->getBlockHeight(this->currentLevel);
    }
    else if (this->currentLevel == 0) {

        int** blockLabels = new int*[this->currentBlockHeightNumber];
        
        for (int i = 0; i < this->currentBlockHeightNumber; ++i) {
            blockLabels[i] = new int[this->currentBlockWidthNumber];
            
            for (int j = 0; j < this->currentBlockWidthNumber; ++j) {
                blockLabels[i][j] = this->currentLabels[i][j];
            }
        }
        
        for (int i = 0; i < this->currentBlockHeightNumber; ++i) {
            
            for (int j = 0; j < this->currentBlockWidthNumber; ++j) {
                
                int heightEnd = this->minimumBlockHeight*i + this->minimumBlockHeight;
                int widthEnd = this->minimumBlockWidth*j + this->minimumBlockWidth;
                
                if (i == this->currentBlockHeightNumber - 1) {
                    heightEnd = this->height;
                }
                
                if (j == this->currentBlockWidthNumber - 1) {
                    widthEnd = this->width;
                }
                
                for (int k = this->minimumBlockHeight*i; k < heightEnd; ++k) {
                    for (int l = this->minimumBlockWidth*j; l < widthEnd; ++l) {
                        this->currentLabels[k][l] = blockLabels[i][j];
                    }
                }
            }
        }
        
        // Remember to free temporary labels.
        delete[] blockLabels;
        
        // Pixel level.
        this->currentBlockWidth = 1;
        this->currentBlockHeight = 1;
        
        this->currentBlockWidthNumber = this->width;
        this->currentBlockHeightNumber = this->height;
    }
    
    #ifdef DEBUG
        for (int i = 0; i < this->currentBlockHeightNumber; ++i) {
            for (int j = 0; j < this->currentBlockWidthNumber; ++j) {
                assert(this->currentLabels[i][j] >= 0);
            }
        }
    
        if (this->initializedHistograms == true) {
            int blockWidthNumber;
            int blockHeightNumber;

//            int blockWidth;
//            int blockHeight;

            int sum = 0;
            for (int level = 1; level <= this->numberOfLevels; ++level) {
//                blockWidth = this->getBlockWidth(level);
//                blockHeight = this->getBlockHeight(level);

                blockWidthNumber = this->getBlockWidthNumber(level);
                blockHeightNumber = this->getBlockHeightNumber(level);

                for (int i = 0; i < blockHeightNumber; ++i) {
                    for (int j = 0; j < blockWidthNumber; ++j) {
                        sum = 0;
                        for (int k = 0; k < this->histogramSize; ++k) {
                            sum += this->histograms[level - 1][i][j][k];
                        }

//                        if (level < this->numberOfLevels) {
//                            assert(this->pixels[level - 1][i][j] >= blockWidth*blockHeight);
//                        }
                        
                        assert(this->pixels[level - 1][i][j] == sum);
                    }
                }
            }
        }
    #endif
}

void SEEDSRevised::initializeHistograms() {
    
    this->histogramDimensions = this->image->channels();
    this->histogramSize = (int) pow(this->numberOfBins, this->histogramDimensions);

    #ifdef UNIFORM
        int denominator = ceil(256./((double) this->numberOfBins));
         
        this->histogramBins = new int*[this->height];
        for (int i = 0; i < this->height; ++i) {
            this->histogramBins[i] = new int[this->width];

            for (int j = 0; j < this->width; ++j) {
                this->histogramBins[i][j] = this->histogramSize;
                
                if (this->histogramDimensions == 1) {
                    this->histogramBins[i][j] = this->image->at<unsigned char>(i, j)/denominator;
                }
                else if (this->histogramDimensions == 3) {
                    this->histogramBins[i][j] = this->image->at<cv::Vec3b>(i, j)[0]/denominator + this->numberOfBins*(this->image->at<cv::Vec3b>(i, j)[1]/denominator) + this->numberOfBins*this->numberOfBins*(this->image->at<cv::Vec3b>(i, j)[2]/denominator);
                }

                #ifdef DEBUG
                    assert(this->histogramBins[i][j] < this->histogramSize);
                #endif
            }
        }
    #else
        int** channels = new int*[this->histogramDimensions];
        int count = 0;

        for (int k = 0; k < this->histogramDimensions; ++k) {
            channels[k] = new int[256];

            for (int l = 0; l < 256; ++l) {
                channels[k][l] = 0;
            }
        }

        for (int i = 0; i < this->height; i += 5) {
            for (int j = 0; j < this->width; j += 5) {
                if (this->histogramDimensions == 1) {
                    ++channels[0][this->image->at<unsigned char>(i, j)];
                }
                else if (this->histogramDimensions == 3) {
                    ++channels[0][this->image->at<cv::Vec3b>(i, j)[0]];
                    ++channels[1][this->image->at<cv::Vec3b>(i, j)[1]];
                    ++channels[2][this->image->at<cv::Vec3b>(i, j)[2]];
                }

                ++count;
            }
        }

        // Compute integral arrays.
        for (int k = 0; k < this->histogramDimensions; ++k) {
            for (int l = 1; l < 256; ++l) {
                channels[k][l] += channels[k][l - 1];
            }
        }

        int equiHeight = ceil(((double) (count + 1))/((double) this->numberOfBins));
        
        this->histogramBins = new int*[this->height];
        for (int i = 0; i < this->height; ++i) {
            this->histogramBins[i] = new int[this->width];

            for (int j = 0; j < this->width; ++j) {
                this->histogramBins[i][j] = this->histogramSize;
                
                if (this->histogramDimensions == 1) {
                    this->histogramBins[i][j] = channels[0][this->image->at<unsigned char>(i, j)]/equiHeight;
                }
                else if (this->histogramDimensions == 3) {
                    this->histogramBins[i][j] = channels[0][this->image->at<cv::Vec3b>(i, j)[0]]/equiHeight
                            + this->numberOfBins*(channels[1][this->image->at<cv::Vec3b>(i, j)[1]]/equiHeight)
                            + this->numberOfBins*this->numberOfBins*(channels[2][this->image->at<cv::Vec3b>(i, j)[2]]/equiHeight);
                }

                #ifdef DEBUG
                    assert(this->histogramBins[i][j] < this->histogramSize);
                #endif
            }
        }
        
        for (int k = 0; k < this->histogramDimensions; ++k) {
            delete[] channels[k];
        }
        
        delete[] channels;
    #endif

    int minimumBlockHeightNumber = this->getBlockHeightNumber(1);
    int minimumBlockWidthNumber = this->getBlockWidthNumber(1);

    int blockHeightEnd;
    int blockWidthEnd;

    this->histograms = new int***[this->numberOfLevels];
    this->pixels = new int**[this->numberOfLevels];

    this->histograms[0] = new int**[minimumBlockHeightNumber];
    this->pixels[0] = new int*[minimumBlockHeightNumber];
        
    for (int i = 0; i < minimumBlockHeightNumber; ++i) {
        this->histograms[0][i] = new int*[minimumBlockWidthNumber];
        this->pixels[0][i] = new int[minimumBlockWidthNumber];

        for (int j = 0; j < minimumBlockWidthNumber; ++j) {
            this->histograms[0][i][j] = new int[this->histogramSize];
            this->pixels[0][i][j] = 0;

            // Initialize histogram bins.
            for (int k = 0; k < this->histogramSize; ++k) {
                this->histograms[0][i][j][k] = 0;
            }

            // Remember the borders, blockHeightEnd and blockWidthEnd
            // are exclusive indices.
            blockHeightEnd = (i + 1)*this->minimumBlockHeight;
            blockWidthEnd = (j + 1)*this->minimumBlockWidth;

            if (i == minimumBlockHeightNumber - 1) {
                blockHeightEnd = this->height;
            }

            if (j == minimumBlockWidthNumber - 1) {
                blockWidthEnd = this->width;
            }

            #ifdef DEBUG
                assert(blockHeightEnd <= this->height);
                assert(blockWidthEnd <= this->width);
            #endif

            for (int k = i*this->minimumBlockHeight; k < blockHeightEnd; ++k) {
                for (int l = j*this->minimumBlockWidth; l < blockWidthEnd; ++l) {
                    ++this->pixels[0][i][j];
                    ++this->histograms[0][i][j][this->histogramBins[k][l]];
                }
            }
        }
    }

    int blockHeightNumber;
    int blockWidthNumber;
    int blockHeightNumberBelow;
    int blockWidthNumberBelow;

    // Calculate histograms at the higher levels by accumulating the histograms
    // at the levels below. First block level is level 1, so we start with level 2.

    // Remember that the used index in this->histograms is on less than the level number.
    for (int level = 2; level <= this->numberOfLevels; ++level) {
        blockHeightNumber = this->getBlockHeightNumber(level);
        blockWidthNumber = this->getBlockWidthNumber(level);
        blockHeightNumberBelow = this->getBlockHeightNumber(level - 1);
        blockWidthNumberBelow = this->getBlockWidthNumber(level - 1);

        this->histograms[level - 1] = new int**[blockHeightNumber];
        this->pixels[level - 1] = new int*[blockHeightNumber];

        for (int i = 0; i < blockHeightNumber; ++i) {
            this->histograms[level - 1][i] = new int*[blockWidthNumber];
            this->pixels[level - 1][i] = new int[blockWidthNumber];

            for (int j = 0; j < blockWidthNumber; ++j) {
                this->histograms[level - 1][i][j] = new int[this->histogramSize];

                this->pixels[level - 1][i][j] = this->pixels[level - 2][2*i][2*j];
                this->pixels[level - 1][i][j] += this->pixels[level - 2][2*i + 1][2*j];
                this->pixels[level - 1][i][j] += this->pixels[level - 2][2*i][2*j + 1];
                this->pixels[level - 1][i][j] += this->pixels[level - 2][2*i + 1][2*j + 1];

                if (i == blockHeightNumber - 1 && 2*i + 2 < blockHeightNumberBelow) {
                    this->pixels[level - 1][i][j] += this->pixels[level - 2][2*i + 2][2*j];
                    this->pixels[level - 1][i][j] += this->pixels[level - 2][2*i + 2][2*j + 1];
                }

                if (j == blockWidthNumber - 1 && 2*j + 2 < blockWidthNumberBelow) {
                    this->pixels[level - 1][i][j] += this->pixels[level - 2][2*i][2*j + 2];
                    this->pixels[level - 1][i][j] += this->pixels[level - 2][2*i + 1][2*j + 2];
                }

                if (i == blockHeightNumber - 1 && j == blockWidthNumber - 1
                        && 2*i + 2 < blockHeightNumberBelow && 2*j + 2 < blockWidthNumberBelow) {
                    this->pixels[level - 1][i][j] += this->pixels[level - 2][2*i + 2][2*j + 2];
                }

                for (int k = 0; k < this->histogramSize; ++k) {
                    this->histograms[level - 1][i][j][k] = this->histograms[level - 2][2*i][2*j][k];
                    this->histograms[level - 1][i][j][k] += this->histograms[level - 2][2*i + 1][2*j][k];
                    this->histograms[level - 1][i][j][k] += this->histograms[level - 2][2*i][2*j + 1][k];
                    this->histograms[level - 1][i][j][k] += this->histograms[level - 2][2*i + 1][2*j + 1][k];

                    if (i == blockHeightNumber - 1 && 2*i + 2 < blockHeightNumberBelow) {
                        this->histograms[level - 1][i][j][k] += this->histograms[level - 2][2*i + 2][2*j][k];
                        this->histograms[level - 1][i][j][k] += this->histograms[level - 2][2*i + 2][2*j + 1][k];
                    }

                    if (j == blockWidthNumber - 1 && 2*j + 2 < blockWidthNumberBelow) {
                        this->histograms[level - 1][i][j][k] += this->histograms[level - 2][2*i][2*j + 2][k];
                        this->histograms[level - 1][i][j][k] += this->histograms[level - 2][2*i + 1][2*j + 2][k];
                    }

                    if (i == blockHeightNumber - 1 && j == blockWidthNumber - 1
                            && 2*i + 2 < blockHeightNumberBelow && 2*j + 2 < blockWidthNumberBelow) {
                        this->histograms[level - 1][i][j][k] += this->histograms[level - 2][2*i + 2][2*j + 2][k];
                    }

                    #ifdef DEBUG
                        assert(this->histograms[level - 1][i][j][k] <= this->pixels[level - 1][i][j]);
                    #endif
                }
            }
        }
    }

    this->initializedHistograms = true;

    #ifdef DEBUG
    
        int blockWidth;
        int blockHeight;

        int sum = 0;
        for (int level = 1; level <= this->numberOfLevels; ++level) {
            blockWidth = this->getBlockWidth(level);
            blockHeight = this->getBlockHeight(level);

            blockWidthNumber = this->getBlockWidthNumber(level);
            blockHeightNumber = this->getBlockHeightNumber(level);

            for (int i = 0; i < blockHeightNumber; ++i) {
                for (int j = 0; j < blockWidthNumber; ++j) {
                    sum = 0;
                    for (int k = 0; k < this->histogramSize; ++k) {
                        sum += this->histograms[level - 1][i][j][k];
                    }
                    
                    assert(this->pixels[level - 1][i][j] >= blockWidth*blockHeight);
                    assert(this->pixels[level - 1][i][j] == sum);
                }
            }
        }
    #endif
}

void SEEDSRevised::performBlockUpdate(int i, int j) {

    if (this->spatialMemory[i][j] == true) {
        
        #ifdef MEMORY
            // Will be set to true i the case the block is moved.
            this->spatialMemory[i][j] = false;
        #endif
        
        int iPlusOne = std::min(i + 1, this->currentBlockHeightNumber - 1);
        int iMinusOne = std::max(i - 1, 0);
        int jPlusOne = std::min(j + 1, this->currentBlockWidthNumber - 1);
        int jMinusOne = std::max(j - 1, 0);

        int labelFrom = this->currentLabels[i][j];
        int labelVerticalForward = this->currentLabels[iPlusOne][j];
        int labelVerticalBackward = this->currentLabels[iMinusOne][j];
        int labelHorizontalForward = this->currentLabels[i][jPlusOne];
        int labelHorizontalBackward = this->currentLabels[i][jMinusOne];

        if (labelVerticalForward != labelFrom
                || labelVerticalBackward != labelFrom
                || labelHorizontalForward != labelFrom
                || labelHorizontalBackward != labelFrom) {

            int iSuperpixelFrom = this->getSuperpixelIFromLabel(labelFrom);
            int jSuperpixelFrom = this->getSuperpixelJFromLabel(labelFrom);

            int blocks = this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom]/this->pixels[this->currentLevel - 1][i][j];
            if (blocks > this->minimumNumberOfSublabels) {

                float currentScore = this->scoreCurrentBlockSegmentation(i, j, iSuperpixelFrom, jSuperpixelFrom);

                int iBest = i;
                int jBest = j;
                int iSuperpixelBest = iSuperpixelFrom;
                int jSuperpixelBest = jSuperpixelFrom;
                float bestScore = 0.;

                if (labelVerticalForward != labelFrom && !this->checkSplitVerticalForward(i, j, iPlusOne, iMinusOne, jPlusOne, jMinusOne)) {
                    int iSuperpixelTo = this->getSuperpixelIFromLabel(labelVerticalForward);
                    int jSuperpixelTo = this->getSuperpixelJFromLabel(labelVerticalForward);

                    float proposedScore = this->scoreProposedBlockSegmentation(i, j, iSuperpixelTo, jSuperpixelTo);
                    
                    if (proposedScore > currentScore + this->minimumConfidence && proposedScore > bestScore) {
                        iBest = iPlusOne;
                        jBest = j;
                        iSuperpixelBest = iSuperpixelTo;
                        jSuperpixelBest = jSuperpixelTo;
                        bestScore = proposedScore;
                    }
                }

                if (labelVerticalBackward != labelFrom && !this->checkSplitVerticalBackward(i, j, iPlusOne, iMinusOne, jPlusOne, jMinusOne)) {
                    int iSuperpixelTo = this->getSuperpixelIFromLabel(labelVerticalBackward);
                    int jSuperpixelTo = this->getSuperpixelJFromLabel(labelVerticalBackward);
                    
                    float proposedScore = this->scoreProposedBlockSegmentation(i, j, iSuperpixelTo, jSuperpixelTo);
                    
                    if (proposedScore > currentScore + this->minimumConfidence && proposedScore > bestScore) {
                        iBest = iMinusOne;
                        jBest = j;
                        iSuperpixelBest = iSuperpixelTo;
                        jSuperpixelBest = jSuperpixelTo;
                        bestScore = proposedScore;
                    }
                }

                if (labelHorizontalForward != labelFrom && !this->checkSplitHorizontalForward(i, j, iPlusOne, iMinusOne, jPlusOne, jMinusOne)) {
                    int iSuperpixelTo = this->getSuperpixelIFromLabel(labelHorizontalForward);
                    int jSuperpixelTo = this->getSuperpixelJFromLabel(labelHorizontalForward);

                    float proposedScore = this->scoreProposedBlockSegmentation(i, j, iSuperpixelTo, jSuperpixelTo);
                    
                    if (proposedScore > currentScore + this->minimumConfidence && proposedScore > bestScore) {
                        iBest = i;
                        jBest = jPlusOne;
                        iSuperpixelBest = iSuperpixelTo;
                        jSuperpixelBest = jSuperpixelTo;
                        bestScore = proposedScore;
                    }
                }

                if (labelHorizontalBackward != labelFrom && !this->checkSplitHorizontalBackward(i, j, iPlusOne, iMinusOne, jPlusOne, jMinusOne)) {
                    int iSuperpixelTo = this->getSuperpixelIFromLabel(labelHorizontalBackward);
                    int jSuperpixelTo = this->getSuperpixelJFromLabel(labelHorizontalBackward);

                    float proposedScore = this->scoreProposedBlockSegmentation(i, j, iSuperpixelTo, jSuperpixelTo);
                    
                    if (proposedScore > currentScore + this->minimumConfidence && proposedScore > bestScore) {
                        iBest = i;
                        jBest = jMinusOne;
                        iSuperpixelBest = iSuperpixelTo;
                        jSuperpixelBest = jSuperpixelTo;
                        bestScore = proposedScore;
                    }
                }

                if (bestScore > 0) {
                    this->updateBlock(i, j, iBest, jBest, iSuperpixelFrom, jSuperpixelFrom, iSuperpixelBest, jSuperpixelBest, iPlusOne, iMinusOne, jPlusOne, jMinusOne);
                }
            }
        }
    }
}

void SEEDSRevised::performPixelUpdate(int i, int j) {
    
    if (this->spatialMemory[i][j] == true) {
        
        #ifdef MEMORY
            // Will be set to true in the case the pixel is moved.
            this->spatialMemory[i][j] = false;
        #endif
            
        int iPlusOne = std::min(i + 1, this->height - 1);
        int iMinusOne = std::max(i - 1, 0);
        int jPlusOne = std::min(j + 1, this->width - 1);
        int jMinusOne = std::max(j - 1, 0);

        int labelFrom = this->currentLabels[i][j];
        int labelVerticalForward = this->currentLabels[iPlusOne][j];
        int labelVerticalBackward = this->currentLabels[iMinusOne][j];
        int labelHorizontalForward = this->currentLabels[i][jPlusOne];
        int labelHorizontalBackward = this->currentLabels[i][jMinusOne];

        #ifdef DEBUG
            assert(labelVerticalForward >= 0);
            assert(labelVerticalBackward >= 0);
            assert(labelHorizontalForward >= 0);
            assert(labelHorizontalBackward >= 0);
        #endif

        if (labelVerticalForward != labelFrom
                || labelVerticalBackward != labelFrom
                || labelHorizontalForward != labelFrom
                || labelHorizontalBackward != labelFrom) {

            int iSuperpixelFrom = this->getSuperpixelIFromLabel(labelFrom);
            int jSuperpixelFrom = this->getSuperpixelJFromLabel(labelFrom);

            if (this->pixels[this->numberOfLevels - 1][iSuperpixelFrom][jSuperpixelFrom] > this->minimumNumberOfSublabels) {

                float currentScore = this->scoreCurrentPixelSegmentation(i, j, iSuperpixelFrom, jSuperpixelFrom);

                int iBest = i;
                int jBest = j;
                int iSuperpixelBest = iSuperpixelFrom;
                int jSuperpixelBest = jSuperpixelFrom;
                float bestScore = 0.;

                if (labelVerticalForward != labelFrom && !this->checkSplitVerticalForward(i, j, iPlusOne, iMinusOne, jPlusOne, jMinusOne)) {
                    int iSuperpixelTo = this->getSuperpixelIFromLabel(labelVerticalForward);
                    int jSuperpixelTo = this->getSuperpixelJFromLabel(labelVerticalForward);

                    float proposedScore = this->scoreProposedPixelSegmentation(i, j, iSuperpixelTo, jSuperpixelTo);
                    float score = this->scorePixelUpdate(i, j, iPlusOne, j, currentScore, proposedScore);

                    if (score > 0 && score > bestScore) {
                        iBest = iPlusOne;
                        jBest = j;
                        iSuperpixelBest = iSuperpixelTo;
                        jSuperpixelBest = jSuperpixelTo;
                        bestScore = score;
                    }
                }

                if (labelVerticalBackward != labelFrom && !this->checkSplitVerticalBackward(i, j, iPlusOne, iMinusOne, jPlusOne, jMinusOne)) {
                    int iSuperpixelTo = this->getSuperpixelIFromLabel(labelVerticalBackward);
                    int jSuperpixelTo = this->getSuperpixelJFromLabel(labelVerticalBackward);

                    float proposedScore = this->scoreProposedPixelSegmentation(i, j, iSuperpixelTo, jSuperpixelTo);
                    float score = this->scorePixelUpdate(i, j, iMinusOne, j, currentScore, proposedScore);

                    if (score > 0 && score > bestScore) {
                        iBest = iMinusOne;
                        jBest = j;
                        iSuperpixelBest = iSuperpixelTo;
                        jSuperpixelBest = jSuperpixelTo;
                        bestScore = score;
                    }
                }

                if (labelHorizontalForward != labelFrom && !this->checkSplitHorizontalForward(i, j, iPlusOne, iMinusOne, jPlusOne, jMinusOne)) {
                    int iSuperpixelTo = this->getSuperpixelIFromLabel(labelHorizontalForward);
                    int jSuperpixelTo = this->getSuperpixelJFromLabel(labelHorizontalForward);

                    float proposedScore = this->scoreProposedPixelSegmentation(i, j, iSuperpixelTo, jSuperpixelTo);
                    float score = this->scorePixelUpdate(i, j, i, jPlusOne, currentScore, proposedScore);

                    if (score > 0 && score > bestScore) {
                        iBest = i;
                        jBest = jPlusOne;
                        iSuperpixelBest = iSuperpixelTo;
                        jSuperpixelBest = jSuperpixelTo;
                        bestScore = score;
                    }
                }

                if (labelHorizontalBackward != labelFrom && !this->checkSplitHorizontalBackward(i, j, iPlusOne, iMinusOne, jPlusOne, jMinusOne)) {
                    int iSuperpixelTo = this->getSuperpixelIFromLabel(labelHorizontalBackward);
                    int jSuperpixelTo = this->getSuperpixelJFromLabel(labelHorizontalBackward);

                    float proposedScore = this->scoreProposedPixelSegmentation(i, j, iSuperpixelTo, jSuperpixelTo);
                    float score = this->scorePixelUpdate(i, j, i, jMinusOne, currentScore, proposedScore);

                    if (score > 0 && score > bestScore) {
                        iBest = i;
                        jBest = jMinusOne;
                        iSuperpixelBest = iSuperpixelTo;
                        jSuperpixelBest = jSuperpixelTo;
                        bestScore = score;
                    }
                }

                if (bestScore > 0) {
                    this->updatePixel(i, j, iBest, jBest, iSuperpixelFrom, jSuperpixelFrom, iSuperpixelBest, jSuperpixelBest, iPlusOne, iMinusOne, jPlusOne, jMinusOne);
                }
            }
        }
    }
}

void SEEDSRevised::iterate(int iterations) {
    
    while (this->currentLevel > 0) {
        
        this->reinitializeSpatialMemory();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            for (int i = 0; i < this->currentBlockHeightNumber; ++i) {
                for (int j = 0; j < this->currentBlockWidthNumber; ++j) {
                    this->performBlockUpdate(i, j);
                }
            }
        }
        
        this->goDownOneLevel();
    }
        
    this->reinitializeSpatialMemory();
    for (int iteration = 0; iteration < 2*iterations; ++iteration) {
        for (int i = 0; i < this->height; ++i) {
            for (int j = 0; j < this->width; ++j) {
                this->performPixelUpdate(i, j);
            }
        }
    }
}

void SEEDSRevised::reinitializeSpatialMemory() {
    for (int i = 0; i < this->currentBlockHeightNumber; ++i) {
        for (int j = 0; j < this->currentBlockWidthNumber; ++j) {
            this->spatialMemory[i][j] = true;
        }
    }
}

int SEEDSRevised::getLevel() const {
    return this->currentLevel;
}

int** SEEDSRevised::getLabels() const {
    assert(this->initializedLabels);
    
    return this->currentLabels;
}

int SEEDSRevised::getNumberOfSuperpixels() const {
    return this->getBlockHeightNumber(this->numberOfLevels)*this->getBlockWidthNumber(this->numberOfLevels);
}

SEEDSRevisedMeanPixels::SEEDSRevisedMeanPixels(const cv::Mat& image, int numberOfLevels, int minimumBlockWidth, int minimumBlockHeight, int numberOfBins, int neighborhoodSize, float minimumConfidence, float spatialWeight) : SEEDSRevised(image, numberOfBins, numberOfLevels, minimumBlockWidth, minimumBlockHeight, neighborhoodSize, minimumConfidence) {
    assert(spatialWeight >= 0);
    assert(spatialWeight <= 1);
    
    this->initializedMeans = false;
    this->spatialWeight = spatialWeight;
    this->spatialNormalization = 1;
}

SEEDSRevisedMeanPixels::SEEDSRevisedMeanPixels(const cv::Mat& image, int desiredNumberOfSuperpixels, int numberOfBins, int neighborhoodSize, float minimumConfidence, float spatialWeight) : SEEDSRevised(image, desiredNumberOfSuperpixels, numberOfBins, neighborhoodSize, minimumConfidence) {
    assert(spatialWeight >= 0);
    assert(spatialWeight <= 1);
    
    this->initializedMeans = false;
    this->spatialWeight = spatialWeight;
    this->spatialNormalization = 1;
}

SEEDSRevisedMeanPixels::~SEEDSRevisedMeanPixels() {
    
    if (this->initializedMeans == true) {
        
        for (int i = 0; i < this->height; ++i) {
            for (int j = 0; j < this->width; ++j) {
                delete[] this->means[0][i][j];
            }
            
            delete[] this->means[0][i];
        }
        
        delete[] this->means[0];
        
        int superpixelHeightNumber = this->getBlockHeightNumber(this->numberOfLevels);
        int superpixelWidthNumber = this->getBlockWidthNumber(this->numberOfLevels);
        
        for (int i = 0; i < superpixelHeightNumber; ++i) {
            for (int j = 0; j < superpixelWidthNumber; ++j) {
                delete[] this->means[1][i][j];
            }

            delete[] this->means[1][i];
        }
        
        delete[] this->means[1];
        delete[] this->means;
        
        this->initializedMeans = false;
    }
}

void SEEDSRevisedMeanPixels::iterate(int iterations) {
    
    while (this->currentLevel > 0) {
        
        this->reinitializeSpatialMemory();
        for (int iteration = 0; iteration < iterations; ++iteration) {
            for (int i = 0; i < this->currentBlockHeightNumber; ++i) {
                for (int j = 0; j < this->currentBlockWidthNumber; ++j) {
                    this->performBlockUpdate(i, j);
                }
            }
        }
        
        this->goDownOneLevel();
    }
    
    this->initializeMeans();
    this->reinitializeSpatialMemory();
    for (int iteration = 0; iteration < 2*iterations; ++iteration) {
        for (int i = 0; i < this->height; ++i) {
            for (int j = 0; j < this->width; ++j) {
                this->performPixelUpdate(i, j);
            }
        }
    }
}

void SEEDSRevisedMeanPixels::initializeMeans() {
    this->meanDimensions = this->histogramDimensions + 2;
    
    this->means = new float***[2];
    this->means[0] = new float**[this->height];
    this->means[1] = new float**[this->superpixelHeightNumber];
    
    for (int i = 0; i < this->superpixelHeightNumber; ++i) {
        this->means[1][i] = new float*[this->superpixelWidthNumber];
        
        for (int j = 0; j < this->superpixelWidthNumber; ++j) {
            this->means[1][i][j] = new float[this->meanDimensions];
            
            for (int k = 0; k < this->meanDimensions; ++k) {
                this->means[1][i][j][k] = 0;
            }
        }
    }
    
    for (int i = 0; i < this->height; ++i) {
        this->means[0][i] = new float*[this->width];
        
        for (int j = 0; j < this->width; ++j) {
            this->means[0][i][j] = new float[this->meanDimensions];
            
            if (this->histogramDimensions == 1) {
                this->means[0][i][j][0] = this->image->at<unsigned char>(i, j);
            }
            else if (this->histogramDimensions == 3) {
                this->means[0][i][j][0] = this->image->at<cv::Vec3b>(i, j)[0];
                this->means[0][i][j][1] = this->image->at<cv::Vec3b>(i, j)[1];
                this->means[0][i][j][2] = this->image->at<cv::Vec3b>(i, j)[2];
            }
            
            this->means[0][i][j][this->meanDimensions - 2] = j;
            this->means[0][i][j][this->meanDimensions - 1] = i;
            
            int iSuperpixel = this->getSuperpixelIFromLabel(this->currentLabels[i][j]);
            int jSuperpixel = this->getSuperpixelJFromLabel(this->currentLabels[i][j]);
            
            for (int k = 0; k < this->meanDimensions; ++k) {
                this->means[1][iSuperpixel][jSuperpixel][k] += this->means[0][i][j][k];
            }
        }
    }
    
    this->colorNormalization = 255.0f*255.0f*this->histogramDimensions;
    this->spatialNormalization = this->height*this->height + this->width*this->width;
    this->initializedMeans = true;
    
    #ifdef DEBUG
        for (int i = 0; i < superpixelHeightNumber; ++i) {
            for (int j = 0; j < superpixelWidthNumber; ++j) {
                for (int k = 0; k < this->histogramDimensions; ++k) {
                    float mean = this->means[1][i][j][k]/this->pixels[this->numberOfLevels - 1][i][j];
                    assert(mean <= 255);
                }
                
                float mean = this->means[1][i][j][this->meanDimensions - 2]/this->pixels[this->numberOfLevels - 1][i][j];
                assert(mean <= this->width);
                
                mean = this->means[1][i][j][this->meanDimensions - 1]/this->pixels[this->numberOfLevels - 1][i][j];
                assert(mean <= this->height);
            }
        }
    #endif
}

void SEEDSRevisedMeanPixels::setSpatialWeight(float spatialWeight) {
    assert(spatialWeight >= 0);
    assert(spatialWeight <= 1);
    
    this->spatialWeight = spatialWeight;
}
