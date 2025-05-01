#include "feature_matching.h"
#include <memory>
int main()
{
    std::unique_ptr<FeatureMatcher> feature_matcher = std::make_unique<FeatureMatcher>();
    feature_matcher->load_model();

    feature_matcher->match("assets/lm81-1.png", 512, "assets/lm81-2.png", 512);
    feature_matcher->match("assets/lm81-1.png", 512, "assets/lm81-2.png", 512);
    feature_matcher->match("assets/lm81-1.png", 512, "assets/lm81-2.png", 512);

    feature_matcher->match("assets/image0.png", 512, "assets/image1.png", 512);
    const int iterations = 100;
    double total_time = 0.0;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        int remain = i % 9;

        switch (remain) {
            case 0:
                feature_matcher->match("assets/lm81-1.png", 512, "assets/lm81-2.png", 512);
                break;
            case 1:
                feature_matcher->match("assets/image0.png", 512, "assets/image1.png", 512);
                break;
            case 2:
                feature_matcher->match("assets/ho_guom1.png", 512, "assets/ho_guom2.png", 512);
                break;
            case 3:
                feature_matcher->match("assets/ho_guom1.png", 512, "assets/ho_guom3.png", 512);
                break;
            case 4:
                feature_matcher->match("assets/ho_guom1.png", 512, "assets/ho_guom4.png", 512);
                break;
            case 5:
                feature_matcher->match("assets/ho_guom2.png", 512, "assets/ho_guom3.png", 512);
                break;
            case 6:
                feature_matcher->match("assets/ho_guom2.png", 512, "assets/ho_guom4.png", 512);
                break;
            case 7:
                feature_matcher->match("assets/ho_guom3.png", 512, "assets/ho_guom4.png", 512);
                break;
            case 8:
                feature_matcher->match("assets/ho_guom4.png", 512, "assets/ho_guom4.png", 512);
                break;
        }

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        total_time += elapsed.count();
    }

    double average_time = total_time / iterations;
    std::cout << "Average time taken: " << average_time << " ms" << std::endl;

    return 0;
}
