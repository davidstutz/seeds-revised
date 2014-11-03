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
        ("process", "show additional information while processing")
        ("csv", "save segmentation as CSV file")
        ("contour", "save contour image of segmentation")
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
    
    bool process = false;
    if (parameters.find("process") != parameters.end()) {
        process = true;
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
                
                if (process == true) {
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
        cv::Mat image = cv::imread(iterator->string(), CV_LOAD_IMAGE_COLOR);
        
        SEEDSRevisedMeanPixels seeds(image, superpixels, numberOfBins, neighborhoodSize, minimumConfidence, spatialWeight);

        timer.restart();
        seeds.initialize();
        seeds.iterate(iterations);
        totalTime += timer.elapsed();
        
        if (process == true) {
            std::cout << Integrity::countSuperpixels(seeds.getLabels(), image.rows, image.cols) << " superpixels for " << iterator->string() << " seconds ..." << std::endl;
        }

        if (parameters.find("contour") != parameters.end()) {

            boost::filesystem::path extension = iterator->filename().extension();
            int position = iterator->filename().string().find(extension.string());
            std::string store = outputDir.string() + DIRECTORY_SEPARATOR + iterator->filename().string().substr(0, position) + "_contours.png";

            int bgr[] = {0, 0, 204};
            cv::Mat contourImage = Draw::contourImage(seeds.getLabels(), image, bgr);
            cv::imwrite(store, contourImage);

            if (process == true) {
                std::cout << "Image " << iterator->string() << " with contours saved to " << store << " ..." << std::endl;
            }
        }

        if (parameters.find("mean") != parameters.end()) {

            boost::filesystem::path extension = iterator->extension();
            int position = iterator->filename().string().find(extension.string());
            std::string store = outputDir.string() + DIRECTORY_SEPARATOR + iterator->filename().string().substr(0, position) + "_mean.png";

            cv::Mat meanImage = Draw::meanImage(seeds.getLabels(), image);
            cv::imwrite(store, meanImage);

            if (process == true) {
                std::cout << "Image " << iterator->string() << " with mean colors saved to " << store << " ..." << std::endl;
            }
        }

        if (parameters.find("csv") != parameters.end()) {

            boost::filesystem::path extension = iterator->extension();
            int position = iterator->filename().string().find(extension.string());
            boost::filesystem::path csvFile(outputDir.string() + DIRECTORY_SEPARATOR + iterator->filename().string().substr(0, position) + ".csv");
            Export::CSV(seeds.getLabels(), image.rows, image.cols, csvFile);

            if (process == true) {
                std::cout << "Labels for image " << iterator->string() << " saved in " << csvFile.string() << " ..." << std::endl;
            }
        }
    }
    
    std::cout << "On average, " << totalTime/images.size() << " seconds needed ..." << std::endl;
    
    return 0;
}