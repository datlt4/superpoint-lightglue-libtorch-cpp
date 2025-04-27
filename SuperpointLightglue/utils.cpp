#include "utils.h"

using namespace FeatureMatching;

// Define resize function similar to Python's
std::pair<cv::Mat, scale_t> FeatureMatching::resize_image(const cv::Mat &image, int size)
{
    int h = image.size().height;
    int w = image.size().width;

    if (!std::isnan(size))
    {
        float scale = static_cast<float>(size) / std::max(h, w);

        int h_new = round(h * scale);
        int w_new = round(w * scale);
        scale_t scale_ = {h_new / float(h), w_new / float(w)};

        cv::Mat resized_image;
        cv::resize(image, resized_image, {w_new, h_new}, 0, 0, cv::INTER_AREA);
        return std::make_pair(resized_image, scale_);
    }
    else
    {
        throw std::invalid_argument("Incorrect new size: " + std::to_string(size));
    }
}

torch::Tensor FeatureMatching::numpy_image_to_torch(const cv::Mat &image)
{
    // Ensure the image is a single-channel 8-bit image (grayscale)
    CV_Assert(image.type() == CV_8UC1);

    // Convert cv::Mat to float32 and normalize to [0,1]
    cv::Mat float_img;
    image.convertTo(float_img, CV_32F, 1.0 / 255.0);

    // Create a tensor from the float image and add batch & channel dimensions
    torch::Tensor tensor_image = torch::from_blob(float_img.data, {1, 1, float_img.rows, float_img.cols}, torch::kFloat32).clone(); // clone to own the memory

    return tensor_image;
}

cv::Mat FeatureMatching::read_image(const std::string &path)
{
    cv::Mat image = cv::imread(path, cv::IMREAD_GRAYSCALE);
    if (image.empty())
    {
        throw std::runtime_error("Could not read image at " + path);
    }

    return image;
}