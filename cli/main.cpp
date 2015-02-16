/**
 * Command line tool for using SEEDS Revised, an implementation of the superpixel
 * algorithm proposed in [1] and evaluated in [2].
 * 
 *  [1] M. van den Bergh, X. Boix, G. Roig, B. de Capitani, L. van Gool.
 *      SEEDS: Superpixels extracted via energy-driven sampling.
 *      Proceedings of the European Conference on Computer Vision, pages 13–26, 2012.
 *  [2] D. Stutz, A. Hermans, B. Leibe.
 *      Superpixel Segmentation using Depth Information.
 *      Bachelor thesis, RWTH Aachen University, Aachen, Germany, 2014.
 * 
 * [2] is available online at 
 * 
 *      http://davidstutz.de/bachelor-thesis-superpixel-segmentation-using-depth-information/
 * 
 * **How to use the command line tool?**
 * 
 * Compile both the library in `/lib` as well as `cli/main.cpp` using CMake (see
 * `README.md`). The provided options can be viewed using 
 * 
 *  $ ./bin/cli --help
 *  Allowed options:
 *   --help                          produce help message
 *   --input arg                     the folder to process, may contain several 
 *                                   images
 *   --bins arg (=5)                 number of bins used for color histograms
 *   --neighborhood arg (=1)         neighborhood size used for smoothing prior
 *   --confidence arg (=0.100000001) minimum confidence used for block update
 *   --iterations arg (=2)           iterations at each level
 *   --spatial-weight arg (=0.25)    spatial weight
 *   --superpixels arg (=400)        desired number of supüerpixels
 *   --verbose                       show additional information while processing
 *   --csv                           save segmentation as CSV file
 *   --contour                       save contour image of segmentation
 *   --labels                        save label image of segmentation
 *   --mean                          save mean colored image of segmentation
 *   --output arg (=output)          specify the output directory (default is 
 *                                   ./output)
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
#include "Tools.h"
#include <opencv2/opencv.hpp>
#include <boost/filesystem.hpp>
#include <boost/timer.hpp>
#include <boost/program_options.hpp>

#if defined(WIN32) || defined(_WIN32)
    #define DIRECTORY_SEPARATOR "\\"
#else
    #define DIRECTORY_SEPARATOR "/"
#endif

int main(int argc, const char** argv) {
    
    boost::program_options::options_description desc("Allowed options");
    desc.add_options()
        ("help", "produce help message")
        ("input", boost::program_options::value<std::string>(), "the folder to process, may contain several images")
        ("bins", boost::program_options::value<int>()->default_value(5), "number of bins used for color histograms")
        ("neighborhood", boost::program_options::value<int>()->default_value(1), "neighborhood size used for smoothing prior")
        ("confidence", boost::program_options::value<float>()->default_value(0.1), "minimum confidence used for block update")
        ("iterations", boost::program_options::value<int>()->default_value(2), "iterations at each level")
        ("spatial-weight", boost::program_options::value<float>()->default_value(0.25), "spatial weight")
        ("superpixels", boost::program_options::value<int>()->default_value(400), "desired number of supüerpixels")
        ("verbose", "show additional information while processing")
        ("csv", "save segmentation as CSV file")
        ("contour", "save contour image of segmentation")
        ("labels", "save label image of segmentation")
        ("mean", "save mean colored image of segmentation")
        ("output", boost::program_options::value<std::string>()->default_value("output"), "specify the output directory (default is ./output)");

    boost::program_options::positional_options_description positionals;
    positionals.add("input", 1);
    
    boost::program_options::variables_map parameters;
    boost::program_options::store(boost::program_options::command_line_parser(argc, argv).options(desc).positional(positionals).run(), parameters);
    boost::program_options::notify(parameters);

    if (parameters.find("help") != parameters.end()) {
        std::cout << desc << std::endl;
        return 1;
    }
    
    boost::filesystem::path outputDir(parameters["output"].as<std::string>());
    if (!boost::filesystem::is_directory(outputDir)) {
        boost::filesystem::create_directory(outputDir);
    }
    
    boost::filesystem::path inputDir(parameters["input"].as<std::string>());
    if (!boost::filesystem::is_directory(inputDir)) {
        std::cout << "Input directory not found ..." << std::endl;
        return 1;
    }
    
    bool verbose = false;
    if (parameters.find("verbose") != parameters.end()) {
        verbose = true;
    }
    
    std::vector<boost::filesystem::path> pathVector;
    std::vector<boost::filesystem::path> images;
    
    std::copy(boost::filesystem::directory_iterator(inputDir), boost::filesystem::directory_iterator(), std::back_inserter(pathVector));

    std::sort(pathVector.begin(), pathVector.end());
    
    std::string extension;
    int count = 0;
    
    for (std::vector<boost::filesystem::path>::const_iterator iterator(pathVector.begin()); iterator != pathVector.end(); ++iterator) {
        if (boost::filesystem::is_regular_file(*iterator)) {
            
            // Check supported file extensions.
            extension = iterator->extension().string();
            std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
            
            if (extension == ".png" || extension == ".jpg" || extension == ".jpeg") {
                images.push_back(*iterator);
                
                if (verbose == true) {
                    std::cout << "Found " << iterator->string() << " ..." << std::endl;
                }
                
                ++count;
            }
        }
    }
    
    std::cout << count << " images total ..." << std::endl;
    
    int iterations = parameters["iterations"].as<int>();
    int numberOfBins = parameters["bins"].as<int>();
    int neighborhoodSize = parameters["neighborhood"].as<int>();
    float minimumConfidence = parameters["confidence"].as<float>();
    float spatialWeight = parameters["spatial-weight"].as<float>();
    int superpixels = parameters["superpixels"].as<int>();
    
    boost::timer timer;
    double totalTime = 0;
    
    for(std::vector<boost::filesystem::path>::iterator iterator = images.begin(); iterator != images.end(); ++iterator) {
        cv::Mat image = cv::imread(iterator->string());
        
        SEEDSRevisedMeanPixels seeds(image, superpixels, numberOfBins, neighborhoodSize, minimumConfidence, spatialWeight);

        timer.restart();
        seeds.initialize();
        seeds.iterate(iterations);
        totalTime += timer.elapsed();
        
        if (verbose == true) {
            std::cout << Integrity::countSuperpixels(seeds.getLabels(), image.rows, image.cols) << " superpixels for " << iterator->string() << " seconds ..." << std::endl;
        }

        if (parameters.find("contour") != parameters.end()) {

            boost::filesystem::path extension = iterator->filename().extension();
            int position = iterator->filename().string().find(extension.string());
            std::string store = outputDir.string() + DIRECTORY_SEPARATOR + iterator->filename().string().substr(0, position) + "_contours.png";

            int bgr[] = {0, 0, 204};
            cv::Mat contourImage = Draw::contourImage(seeds.getLabels(), image, bgr);
            cv::imwrite(store, contourImage);

            if (verbose == true) {
                std::cout << "Image " << iterator->string() << " with contours saved to " << store << " ..." << std::endl;
            }
        }

        if (parameters.find("labels") != parameters.end()) {

            boost::filesystem::path extension = iterator->filename().extension();
            int position = iterator->filename().string().find(extension.string());
            std::string store = outputDir.string() + DIRECTORY_SEPARATOR + iterator->filename().string().substr(0, position) + "_labels.png";
            
            cv::Mat labelImage = Draw::labelImage(seeds.getLabels(), image);
            cv::imwrite(store, labelImage);

            if (verbose == true) {
                std::cout << "Image " << iterator->string() << " with labels saved to " << store << " ..." << std::endl;
            }
        }
        
        if (parameters.find("mean") != parameters.end()) {

            boost::filesystem::path extension = iterator->extension();
            int position = iterator->filename().string().find(extension.string());
            std::string store = outputDir.string() + DIRECTORY_SEPARATOR + iterator->filename().string().substr(0, position) + "_mean.png";

            cv::Mat meanImage = Draw::meanImage(seeds.getLabels(), image);
            cv::imwrite(store, meanImage);

            if (verbose == true) {
                std::cout << "Image " << iterator->string() << " with mean colors saved to " << store << " ..." << std::endl;
            }
        }

        if (parameters.find("csv") != parameters.end()) {

            boost::filesystem::path extension = iterator->extension();
            int position = iterator->filename().string().find(extension.string());
            boost::filesystem::path csvFile(outputDir.string() + DIRECTORY_SEPARATOR + iterator->filename().string().substr(0, position) + ".csv");
            Export::CSV(seeds.getLabels(), image.rows, image.cols, csvFile);

            if (verbose == true) {
                std::cout << "Labels for image " << iterator->string() << " saved in " << csvFile.string() << " ..." << std::endl;
            }
        }
    }
    
    std::cout << "On average, " << totalTime/images.size() << " seconds needed ..." << std::endl;
    
    return 0;
}
