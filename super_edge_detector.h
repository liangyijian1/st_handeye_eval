#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <filesystem>
#include <ceres/ceres.h>
#include <cmath>
#include <fstream>

namespace fs = std::filesystem;

struct RadialProfileSample
{
    double distance;  // distance from the center along the direction
    double intensity; // intensity at the sample point
    double gradient;  // gradient magnitude at the sample point
};

struct detector_config
{
    int num_directions = 16; // number of radial directions to sample
    double radial_margin = 5; // margin around the detected circle radius to sample
    double radial_step = 0.1; // step size for sampling along the radial direction
};

struct AsymmetricGaussianFunctor
{
    AsymmetricGaussianFunctor(double x, double y) : x_(x), y_(y) {}

    template <typename T>
    bool operator()(const T* const a, const T* const mu, const T* const sigma1, const T* const sigma2, T* residual) const
    {
        T sig1 = ceres::abs(sigma1[0]) + T(1e-6);
        T sig2 = ceres::abs(sigma2[0]) + T(1e-6);
        T diff = T(x_) - mu[0];

        T current_sigma = (diff < T(0.0)) ? sig1 : sig2;
        T constant_term = T(2.0) / T(std::sqrt(2.0 * CV_PI));

        T exp_term = ceres::exp(-(diff * diff) / (T(2.0) * current_sigma * current_sigma));
        T f_x = a[0] * constant_term * (T(1.0) / (sig1 + sig2)) * exp_term;
        residual[0] = T(y_) - f_x;
        return true;
    }

private:
    const double x_;
    const double y_;
};

struct AsymmetricGaussianResult
{
    double a = 0.0;
    double mu = 0.0;
    double sigma1 = 0.0;
    double sigma2 = 0.0;
};

class super_edge_detector
{
public:
    super_edge_detector(const std::string& root_path, const detector_config& config)
        : root_path_(root_path), config(config)
    {};
    ~super_edge_detector() {};
    bool detect_edges();

private:
    bool detect_circles(const cv::Mat& image, std::vector<cv::Vec3f>& circles, const std::string& save_path = "");

    bool radial_profile(const cv::Mat& image, const cv::Vec3f& center, const cv::Point2d& direction, const detector_config& config, std::vector<RadialProfileSample>& profile);

    bool cal_profile_gradient(const std::vector<RadialProfileSample>& profile, const detector_config& config, std::vector<double>& gradients, std::vector<double>& x_values);

    bool ceres_optimization(
        const std::vector<double>& gradients, 
        const std::vector<double>& x_values, 
        const detector_config& config, 
        AsymmetricGaussianResult& result,
        ceres::Solver::Summary& summary);

    void draw_subpixel_edges(cv::Mat& image, 
                             const std::vector<cv::Point2d>& edge_points, 
                             const std::vector<int>& ray_indices = {},
                             const std::vector<std::vector<cv::Point2d>>& sample_points = {},
                             bool draw_circles = true, 
                             int shift = 8);

    void plot_fitting_curve(const std::vector<double>& x_values, 
                            const std::vector<double>& gradients, 
                            double a, double mu, double sigma1, double sigma2, 
                            const ceres::Solver::Summary& summary,
                            const std::string& save_path,
                            int ray_index);

    std::string root_path_;
    detector_config config;
};

