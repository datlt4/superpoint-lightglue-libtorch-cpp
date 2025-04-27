#include "lightglue.h"

LightGlue::LightGlue(torch::Device device) : device(device)
{
    // Constructor
}

LightGlue::~LightGlue()
{
    // Destructor
}

void LightGlue::load_model(const std::string &model_path)
{
    try
    {
        this->lightglue_scripted = torch::jit::load(model_path, torch::kCUDA);
    }
    catch (const c10::Error &e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << e.msg() << std::endl;
        std::cerr << "[Error] loading the model\n";
        assert(false);
    }
}

void LightGlue::preprocess(SuperPointFeature &feature0, SuperPointFeature &feature1)
{
    // Preprocess features
    feature0.keypoints = feature0.keypoints.to(this->device);
    feature0.descriptors = feature0.descriptors.to(this->device);
    feature1.keypoints = feature1.keypoints.to(this->device);
    feature1.descriptors = feature1.descriptors.to(this->device);

    // Keypoint normalization
    float kn0 = static_cast<float>(feature0.size) / 2.0f;
    float kn1 = static_cast<float>(feature1.size) / 2.0f;

    torch::Tensor divisor0 = torch::full_like(feature0.keypoints, kn0);
    torch::Tensor divisor1 = torch::full_like(feature1.keypoints, kn1);

    feature0.keypoints = (feature0.keypoints - divisor0) / divisor0;
    feature1.keypoints = (feature1.keypoints - divisor1) / divisor1;
}

void LightGlue::predict(SuperPointFeature& feature0, SuperPointFeature& feature1, Matches &matches)
{
    // Prepare inputs
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(feature0.keypoints);
    inputs.push_back(feature0.descriptors);
    inputs.push_back(feature1.keypoints);
    inputs.push_back(feature1.descriptors);

    // Run the model
    auto output_els = this->lightglue_scripted.forward(inputs).toTuple()->elements();
    torch::Tensor matches01 = output_els[0].toTensor();
    // torch::Tensor mscores01 = output_els[1].toTensor();

    matches.matches.clear();
    
    torch::Tensor matches_squeezed = matches01.cpu();
    auto matches_accessor = matches_squeezed.accessor<long, 2>();
    for (int i = 0; i < matches_squeezed.size(0); ++i)
    {
        int idx0 = static_cast<int>(matches_accessor[i][0]);
        int idx1 = static_cast<int>(matches_accessor[i][1]);
        matches.matches.emplace_back(idx0, idx1, 0.0f);
    }
}