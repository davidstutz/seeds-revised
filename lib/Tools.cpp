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
#include "Tools.h"
#include "SeedsRevised.h"

cv::Mat Draw::contourImage(int** labels, const cv::Mat &image, int* bgr) {
    
    cv::Mat newImage = image.clone();
    
    int label = 0;
    int labelTop = -1;
    int labelBottom = -1;
    int labelLeft = -1;
    int labelRight = -1;
    
    for (int i = 0; i < newImage.rows; i++) {
        for (int j = 0; j < newImage.cols; j++) {
            
            label = labels[i][j];
            
            labelTop = label;
            if (i > 0) {
                labelTop = labels[i - 1][j];
            }
            
            labelBottom = label;
            if (i < newImage.rows - 1) {
                labelBottom = labels[i + 1][j];
            }
            
            labelLeft = label;
            if (j > 0) {
                labelLeft = labels[i][j - 1];
            }
            
            labelRight = label;
            if (j < newImage.cols - 1) {
                labelRight = labels[i][j + 1];
            }
            
            if (label != labelTop || label != labelBottom || label!= labelLeft || label != labelRight) {
                newImage.at<cv::Vec3b>(i, j)[0] = bgr[0];
                newImage.at<cv::Vec3b>(i, j)[1] = bgr[1];
                newImage.at<cv::Vec3b>(i, j)[2] = bgr[2];
            }
        }
    }
    
    return newImage;
}

cv::Mat Draw::meanImage(int** labels, const cv::Mat &image) {
    assert(image.channels() == 3);
    
    cv::Mat newImage = image.clone();
    int numberOfSuperpixels = Integrity::countSuperpixels(labels, newImage.rows, newImage.cols);
    
    int meanB = 0;
    int meanG = 0;
    int meanR = 0;
    int count = 0;
    
    for (int label = 0; label < numberOfSuperpixels; label++) {
        meanB = 0;
        meanG = 0;
        meanR = 0;
        count = 0;
        
        for (int i = 0; i < newImage.rows; i++) {
            for (int j = 0; j < newImage.cols; j++) {
                if (labels[i][j] == label) {
                    meanB += image.at<cv::Vec3b>(i, j)[0];
                    meanG += image.at<cv::Vec3b>(i, j)[1];
                    meanR += image.at<cv::Vec3b>(i, j)[2];
                    
                    count++;
                }
            }
        }
        
        if (count > 0) {
            meanB = (int) meanB/count;
            meanG = (int) meanG/count;
            meanR = (int) meanR/count;
        }
        
        for (int i = 0; i < newImage.rows; i++) {
            for (int j = 0; j < newImage.cols; j++) {
                if (labels[i][j] == label) {
                    newImage.at<cv::Vec3b>(i, j)[0] = meanB;
                    newImage.at<cv::Vec3b>(i, j)[1] = meanG;
                    newImage.at<cv::Vec3b>(i, j)[2] = meanR;
                }
            }
        }
    }
    
    return newImage;
}

cv::Mat Draw::labelImage(int** labels, const cv::Mat &image) {
    cv::Mat newImage = image.clone();
    
    int maxLabel = 0;
    for (int i = 0; i < newImage.rows; i++) {
        for (int j = 0; j < newImage.cols; j++) {
            // assert(labels[i][j] >= 0);
            
            if (labels[i][j] > maxLabel) {
                maxLabel = labels[i][j];
            }
        }
    }
    
    // Always add 1 to allow -1 as label.
    int** colors = new int*[maxLabel + 2];
    for (int k = 0; k < maxLabel + 2; ++k) {
        colors[k] = new int[3];
        colors[k][0] = rand() % 256;
        colors[k][1] = rand() % 256;
        colors[k][2] = rand() % 256;
    }
    
    for (int i = 0; i < newImage.rows; ++i) {
        for (int j = 0; j < newImage.cols; ++j) {
            if (labels[i][j] >= 0) {
                int label = labels[i][j];
                newImage.at<cv::Vec3b>(i, j)[0] = colors[label + 1][0];
                newImage.at<cv::Vec3b>(i, j)[1] = colors[label + 1][1];
                newImage.at<cv::Vec3b>(i, j)[2] = colors[label + 1][2];
            }
            else {
                newImage.at<cv::Vec3b>(i, j)[0] = 0;
                newImage.at<cv::Vec3b>(i, j)[1] = 0;
                newImage.at<cv::Vec3b>(i, j)[2] = 0;
            }
        }
    }
    
    for (int k = 0; k < maxLabel + 1; ++k) {
        delete[] colors[k];
    }
    delete[] colors;
    
    return newImage;
}

int Integrity::countSuperpixels(int** labels, int rows, int cols) {
    assert(rows > 0);
    assert(cols > 0);
    
    int maxLabel = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            assert(labels[i][j] >= 0);
            
            if (labels[i][j] > maxLabel) {
                maxLabel = labels[i][j];
            }
        }
    }
    
    bool* foundLabels = new bool[maxLabel + 1];
    for (int k = 0; k < maxLabel + 1; k++) {
        foundLabels[k] = false;
    }
    
    int count = 0;
    int label = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            label = labels[i][j];
            
            if (foundLabels[label] == false) {
                foundLabels[label] = true;
                count++;
            }
        }
    }
    
    // Remember to free.
    delete[] foundLabels;
    
    return count;
}

void Integrity::relabel(int** labels, int rows, int cols) {
    assert(rows > 0);
    assert(cols > 0);
    
    int maxLabel = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (labels[i][j] > maxLabel) {
                maxLabel = labels[i][j];
            }
        }
    }
    
    int* relabeling = new int[maxLabel + 1];
    for (int l = 0; l < maxLabel + 1; ++l) {
        relabeling[l] = -1;
    }
    
    int label = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (relabeling[labels[i][j]] < 0) {
                relabeling[labels[i][j]] = label;
                ++label;
            }
            
            labels[i][j] = relabeling[labels[i][j]];
        }
    }
    
    delete[] relabeling;
}

void Export::CSV(int** labels, int rows, int cols, boost::filesystem::path path) {
    assert(rows > 0);
    assert(cols > 0);
    
    boost::filesystem::fstream csvFile;
    csvFile.open(path.c_str(), boost::filesystem::ofstream::out);
    
    assert(csvFile);
    
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            csvFile << labels[i][j];
            
            if (j < cols - 1) {
                csvFile << ",";
            }
        }
        
        csvFile << "\n";
    }
    
    csvFile.close();
}

template <typename T>
void Export::BSDEvaluationFile(const cv::Mat &matrix, int precision, boost::filesystem::path path) {
    boost::filesystem::fstream file;
    file.open(path.c_str(), boost::filesystem::ofstream::out);
    
    assert(file);
    file.precision(precision);
    
    for (int i = 0; i < matrix.rows; i++) {
        for (int j = 0; j < matrix.cols; j++) {
            
            T value = matrix.at<T>(i, j);
            
            int order = 10;
            int fill = precision + 2;
            while (abs(value) >= order) {
                ++fill;
                order *= 10;
            }
            
            assert(fill >= 0);
            assert(fill <= 10);
            
            for (int k = 0; k < 10 - fill; ++k) {
                file << " ";
            }
            
            file << matrix.at<T>(i, j);
            
            if (j < matrix.cols - 1) {
                file << " ";
            }
        }
        
        file << "\n";
    }
    
    file.close();
}

template void Export::BSDEvaluationFile<double>(const cv::Mat&, int, boost::filesystem::path);