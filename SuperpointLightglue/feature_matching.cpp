#include "feature_matching.h"

FeatureMatcher::FeatureMatcher(torch::Device device) : device(device)
{
    // Constructor
    superpoint = SuperPoint(this->device);
    lightglue = LightGlue(this->device);
}
FeatureMatcher::FeatureMatcher() : device(torch::kCPU)
{
    if (torch::cuda::is_available())
    {
        std::cout << "CUDA is available! " << std::endl;
        device = torch::kCUDA;
    }
    else
    {
        std::cout << "CUDA is not available, using CPU." << std::endl;
    }
    superpoint = SuperPoint(this->device);
    lightglue = LightGlue(this->device);
}

FeatureMatcher::~FeatureMatcher()
{
    // Destructor
}

void FeatureMatcher::load_model()
{
    const std::string superpoint_model_path = "weights/superpoint_scripted.pt";
    superpoint.load_model(superpoint_model_path);
    const std::string lightglue_model_path = "weights/lightglue_scripted.pt";
    lightglue.load_model(lightglue_model_path);
    std::cout << "Model loaded successfully!" << std::endl;
}

void FeatureMatcher::extract_features(const cv::Mat &image, unsigned int size, SuperPointFeature &features)
{
    // Preprocess images
    std::pair<torch::Tensor, scale_t> processed_input = superpoint.preprocess(image, size);
    torch::Tensor input_tensor = processed_input.first;
    scale_t scale = processed_input.second;
    superpoint.predict(input_tensor, features);
    features.scale = scale;
    features.size = size;
}

void FeatureMatcher::extract_features(const std::string &image_path, unsigned int size, SuperPointFeature &features)
{
    cv::Mat image = read_image(image_path);
    this->extract_features(image, size, features);
}

void FeatureMatcher::match(SuperPointFeature &feature0, SuperPointFeature &feature1, Matches &matches)
{
    // Preprocess features
    this->lightglue.preprocess(feature0, feature1);
    this->lightglue.predict(feature0, feature1, matches);
}

void FeatureMatcher::match(const cv::Mat &image0, unsigned int size0, const cv::Mat &image1, unsigned int size1, bool draw_matches)
{
    // Preprocess images
    SuperPointFeature features0, features1;
    Matches matches;
    this->extract_features(image0, size0, features0);
    convert_to_cv_kpts(features0, matches.keypoints0);
    this->extract_features(image1, size1, features1);
    convert_to_cv_kpts(features1, matches.keypoints1);
    this->match(features0, features1, matches);

    // Draw matches
    if (draw_matches)
    {
        cv::Mat image_matches;
        cv::drawMatches(image0, matches.keypoints0, image1, matches.keypoints1, matches.matches, image_matches);
        cv::imwrite("Matches.jpg", image_matches);
    }
    else
    {
        std::cout << "Matches: " << matches.matches.size() << std::endl;
    }
}

void FeatureMatcher::match(const std::string &image_path0, unsigned int size0, const std::string &image_path1, unsigned int size1)
{
    cv::Mat image0 = read_image(image_path0);
    cv::Mat image1 = read_image(image_path1);
    this->match(image0, size0, image1, size1, true);
}

void convert_to_cv_kpts(const SuperPointFeature &features, std::vector<cv::KeyPoint> &keypoints)
{
    keypoints.clear();
    // Squeeze batch dim: (1, 512, 2) -> (512, 2)
    torch::Tensor keypoints_squeezed = features.keypoints.squeeze(0).contiguous().cpu();
    auto keypoints_accessor = keypoints_squeezed.accessor<float, 2>();
    for (int i = 0; i < keypoints_squeezed.size(0); ++i)
    {
        float x = keypoints_accessor[i][0];
        float y = keypoints_accessor[i][1];
        float s_h = features.scale.scale_h;
        float s_w = features.scale.scale_w;
        keypoints.emplace_back(x / s_h, y / s_w, 1);
    }
}
