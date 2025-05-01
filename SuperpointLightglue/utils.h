#pragma once
#ifndef __FEATURE_MATCHING_UTILS_H__
#define __FEATURE_MATCHING_UTILS_H__

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp>
#include <cmath>

namespace FeatureMatching
{
    typedef struct scale
    {
        float scale_h = 1;
        float scale_w = 1;
    } scale_t;

    struct SuperPointFeature
    {
        torch::Tensor keypoints;
        torch::Tensor scores;
        torch::Tensor descriptors;
        scale_t scale;
        cv::Size size;

        // Convert to PyTorch tensor
        torch::Tensor size_to_tensor() const
        {
            std::vector<int> data = {size.width, size.height};
            return torch::from_blob(data.data(), {2}, torch::kFloat32);
        }
    };

    struct Matches
    {
        std::vector<cv::DMatch> matches;
        std::vector<cv::KeyPoint> keypoints0;
        std::vector<cv::KeyPoint> keypoints1;

        Matches()
        {
            matches.reserve(512);
            keypoints0.reserve(512);
            keypoints1.reserve(512);
        }
    };

    // Define resize function similar to Python's
    std::pair<cv::Mat, scale_t> resize_image(const cv::Mat &image, int size);

    torch::Tensor numpy_image_to_torch(const cv::Mat &image);

    cv::Mat read_image(const std::string &path, bool grayscale = false);
}

#endif // __FEATURE_MATCHING_UTILS_H__
