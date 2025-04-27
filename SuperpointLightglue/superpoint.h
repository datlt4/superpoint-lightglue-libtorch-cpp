#pragma once
#ifndef __SUPERPOINT_H__
#define __SUPERPOINT_H__
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <assert.h>
#include <map>
#include <chrono>
#include <atomic>
#include <fstream>

#include "utils.h"

using namespace FeatureMatching;

class SuperPoint
{
public:
    SuperPoint(torch::Device device = torch::kCPU);
    ~SuperPoint();
    void load_model(const std::string& model_path);
    std::pair<torch::Tensor, scale_t> preprocess(const std::string &path, int resize = 512);
    std::pair<torch::Tensor, scale_t> preprocess(const cv::Mat &input, int resize = 512);
    void predict(torch::Tensor& input_tensor, SuperPointFeature& output_features);

private:
    torch::Device device;
    torch::jit::script::Module superpoint_scripted;
};

#endif // __SUPERPOINT_H__