#pragma once
#ifndef __FEATURE_MATCHING_H__
#define __FEATURE_MATCHING_H__

#include "utils.h"
#include "superpoint.h"
#include "lightglue.h"

void convert_to_cv_kpts(const SuperPointFeature &features, std::vector<cv::KeyPoint> &keypoints);

class FeatureMatcher
{
public:
    FeatureMatcher(torch::Device device);
    FeatureMatcher();
    ~FeatureMatcher();

    // void match(const cv::Mat &image0, unsigned int size, const cv::Mat &image1, unsigned int size1, const std::string &output_path);
    void load_model();
    void extract_features(const cv::Mat &image, unsigned int size, SuperPointFeature &features);
    void extract_features(const std::string &image_path, unsigned int size, SuperPointFeature &features);

    void match(SuperPointFeature &feature0, SuperPointFeature &feature1, Matches &matches);
    void match(const cv::Mat &image0, unsigned int size0, const cv::Mat &image1, unsigned int size1, bool draw_matches = false);
    void match(const std::string &image_path0, unsigned int size0, const std::string &image_path1, unsigned int size1);

private:
    torch::Device device;
    SuperPoint superpoint;
    LightGlue lightglue;
};

#endif // __FEATURE_MATCHING_H__
