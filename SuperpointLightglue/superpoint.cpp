#include "superpoint.h"

SuperPoint::SuperPoint(torch::Device device) : device(device)
{
    // Constructor
}

SuperPoint::~SuperPoint()
{
    // Destructor
}

void SuperPoint::load_model(const std::string &model_path)
{
    try
    {
        this->superpoint_scripted = torch::jit::load(model_path, torch::kCUDA);
    }
    catch (const c10::Error &e)
    {
        std::cerr << e.what() << std::endl;
        std::cerr << e.msg() << std::endl;
        std::cerr << "[Error] loading the model\n";
        assert(false);
    }
}

// Main function that reads an image and converts it to a PyTorch tensor
std::pair<torch::Tensor, scale_t> SuperPoint::preprocess(const std::string &path, int resize)
{
    cv::Mat image = read_image(path);

    scale_t scale = {1.0, 1.0};

    if (resize != 0)
    {
        std::pair<cv::Mat, scale_t> resized_pair = resize_image(image, resize);
        image = resized_pair.first;
        scale = resized_pair.second;
    }

    return std::make_pair(numpy_image_to_torch(image).to(this->device), scale);
}

std::pair<torch::Tensor, scale_t> SuperPoint::preprocess(const cv::Mat &input, int resize)
{
    cv::Mat image = input.clone();

    if (image.empty())
    {
        throw std::runtime_error("Input image is empty");
    }
    if (image.channels() == 3)
    {
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    }
    else if (image.channels() == 4)
    {
        cv::cvtColor(image, image, cv::COLOR_BGRA2GRAY);
    }
    else if (image.channels() != 1)
    {
        throw std::runtime_error("Input image must be grayscale");
    }

    scale_t scale = {1.0, 1.0};

    if (resize != 0)
    {
        std::pair<cv::Mat, scale_t> resized_pair = resize_image(image, resize);
        image = resized_pair.first;
        scale = resized_pair.second;
    }
    return std::make_pair(numpy_image_to_torch(image).to(this->device), scale);
}

void SuperPoint::predict(torch::Tensor &input_tensor, SuperPointFeature &output_features)
{
    // Forward pass through the model
    auto output_els = this->superpoint_scripted.forward({input_tensor}).toTuple()->elements();

    output_features.keypoints = output_els[0].toTensor();   // keypoints
    output_features.scores = output_els[1].toTensor();      // scores
    output_features.descriptors = output_els[2].toTensor(); // descriptors
}
