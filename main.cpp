#include "feature_matching.h"
#include <memory>
int main()
{
    std::unique_ptr<FeatureMatcher> feature_matcher = std::make_unique<FeatureMatcher>();
    feature_matcher->load_model();

    // feature_matcher->match("assets/lm81-1.png", 512, "assets/lm81-2.png", 512);
    // feature_matcher->match("assets/lm81-1.png", 512, "assets/lm81-2.png", 512);
    // feature_matcher->match("assets/lm81-1.png", 512, "assets/lm81-2.png", 512);

    // const int iterations = 100;
    // double total_time = 0.0;

    // for (int i = 0; i < iterations; ++i) {
    //     auto start = std::chrono::high_resolution_clock::now();

        // feature_matcher->match("assets/lm81-1.png", 512, "assets/lm81-2.png", 512);
        feature_matcher->match("assets/google_map.png", 512, "assets/satelite_rotate.png", 1024);

    //     auto end = std::chrono::high_resolution_clock::now();
    //     std::chrono::duration<double, std::milli> elapsed = end - start;
    //     total_time += elapsed.count();
    // }

    // double average_time = total_time / iterations;
    // std::cout << "Average time taken: " << average_time << " ms" << std::endl;

    return 0;
}
