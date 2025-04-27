#pragma once
#ifndef __LIGHTGLUE_H__
#define __LIGHTGLUE_H__
#include <torch/script.h>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <tuple>
#include "utils.h"

using namespace FeatureMatching;

class LightGlue
{
public:
    LightGlue(torch::Device device = torch::kCPU);
    ~LightGlue();
    void load_model(const std::string &model_path);
    void preprocess(SuperPointFeature &feature0, SuperPointFeature &feature1);
    void predict(SuperPointFeature &feature0, SuperPointFeature &feature1, Matches &matches);

private:
    torch::Device device;
    torch::jit::script::Module lightglue_scripted;
};

#endif // __LIGHTGLUE_H__
