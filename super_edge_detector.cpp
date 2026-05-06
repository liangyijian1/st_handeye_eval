#include "super_edge_detector.h"

#include <algorithm>
#include <cmath>
#include <iostream>


bool makeGray64F(const cv::Mat& image, cv::Mat& gray64)
{
    if (image.empty()) {
        return false;
    }

    cv::Mat gray;
    if (image.channels() == 1) {
        if (image.depth() == CV_64F) {
            gray64 = image;
            return true;
        }
        gray = image;
    }
    else if (image.channels() == 3) {
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    }
    else if (image.channels() == 4) {
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    }
    else {
        return false;
    }

    gray.convertTo(gray64, CV_64F);
    return true;
}

bool bilinearSample(const cv::Mat& gray64, const cv::Point2d& point, double& value)
{
    if (point.x < 0.0 || point.y < 0.0 ||
        point.x > static_cast<double>(gray64.cols - 1) ||
        point.y > static_cast<double>(gray64.rows - 1)) {
        return false;
    }

    const int x0 = static_cast<int>(std::floor(point.x));
    const int y0 = static_cast<int>(std::floor(point.y));
    const int x1 = std::min(x0 + 1, gray64.cols - 1);
    const int y1 = std::min(y0 + 1, gray64.rows - 1);
    const double dx = point.x - static_cast<double>(x0);
    const double dy = point.y - static_cast<double>(y0);

    const double v00 = gray64.at<double>(y0, x0);
    const double v10 = gray64.at<double>(y0, x1);
    const double v01 = gray64.at<double>(y1, x0);
    const double v11 = gray64.at<double>(y1, x1);

    value = (1.0 - dx) * (1.0 - dy) * v00 +
        dx * (1.0 - dy) * v10 +
        (1.0 - dx) * dy * v01 +
        dx * dy * v11;
    return std::isfinite(value);
}


bool super_edge_detector::detect_edges()
{
    // 1. Load images from root_path_
    auto root_path = std::filesystem::path(root_path_);
    for (const auto& entry : std::filesystem::directory_iterator(root_path))
    {
        if (entry.is_regular_file())
        {
            const auto& path = entry.path();
            if (path.extension() == ".bmp")
            {
                cv::Mat image = cv::imread(path.string());
                if (image.empty())
                {
                    std::cerr << "Failed to load image: " << path << std::endl;
                    continue;
                }
                // 2. find all circles in the image
                std::vector<cv::Vec3f> circles;
                detect_circles(image, circles);
                std::cout << "Detected " << circles.size() << " circles in image: " << path << std::endl;
                cv::Mat displayImage = image.clone();
                cv::Mat profileImage;
                if (!makeGray64F(image, profileImage)) {
                    std::cerr << "Failed to build radial profile image: " << path << std::endl;
                    continue;
                }
                
                for (size_t i = 0; i < circles.size(); i++)
                {
                    std::vector<cv::Point2d> edgePoints;
                    for (size_t j = 0; j < config.num_directions; j++)
                    {
                        double angle = 2.0 * CV_PI * static_cast<double>(j) / static_cast<double>(config.num_directions);
                        const cv::Point2d direction(std::cos(angle), std::sin(angle));

                        std::vector<RadialProfileSample> profile;
                        // 3. radial profile
                        radial_profile(profileImage, circles[i], direction, config, profile);
                        
                        // 4. profile gradient
                        std::vector<double> gradients;
                        std::vector<double> x_values;
                        cal_profile_gradient(profile, config, gradients, x_values);

                        // 5.ceres optimization
                        double optimized_value = 0.0;
                        ceres_optimization(gradients, x_values, config, optimized_value);

                        // 6. map edge position back to image
                        edgePoints.emplace_back(circles[i][0] + optimized_value * direction.x, circles[i][1] + optimized_value * direction.y);
                    }
                    
                    // 7. draw detected edges on the image and save
                    draw_subpixel_edges(displayImage, edgePoints, true, 8);
                }
                cv::imwrite(path.stem().string() + "_edges.jpg", displayImage);
            }
        }
    }
    return false;
}

bool super_edge_detector::detect_circles(const cv::Mat& image, std::vector<cv::Vec3f>& circles)
{
    if (image.empty())
        return false;
    cv::Mat display = image.clone();
    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image.clone();

    cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);
    std::vector<cv::Vec3f> detected;
    cv::HoughCircles(gray, detected, cv::HOUGH_GRADIENT,
                     1,                // dp
                     gray.rows / 16,   // minDist
                     150,              // param1
                     30,               // param2
                     2,               // minRadius
                     15);               // maxRadius

    circles = detected;

    for (const auto& c : circles)
    {
        cv::Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);
        // circle profile
        cv::circle(display, center, radius, cv::Scalar(0, 255, 0), 2);
        cv::circle(display, center, 3, cv::Scalar(0, 0, 255), -1);
    }
    cv::imwrite("detected_circles.jpg", display);
    return true;
}

bool super_edge_detector::radial_profile(const cv::Mat &image, const cv::Vec3f &center, const cv::Point2d &direction, const detector_config &config, std::vector<RadialProfileSample> &profile)
{
    const double radius = static_cast<double>(center[2]);
    const double directionNorm = std::sqrt(direction.x * direction.x + direction.y * direction.y);
    if (!std::isfinite(radius) || radius <= 0.0 ||
        !std::isfinite(directionNorm) || directionNorm <= 0.0 ||
        !std::isfinite(config.radial_margin) || config.radial_margin <= 0.0 ||
        !std::isfinite(config.radial_step) || config.radial_step <= 0.0) {
        return false;
    }

    cv::Mat gray64;
    if (!makeGray64F(image, gray64)) {
        return false;
    }

    const cv::Point2d circleCenter(static_cast<double>(center[0]), static_cast<double>(center[1]));
    const cv::Point2d unitDirection(direction.x / directionNorm, direction.y / directionNorm);
    const double startDistance = std::max(0.0, radius - config.radial_margin);
    const double endDistance = radius + config.radial_margin;

    for (double distance = startDistance; distance <= endDistance + 1e-9; distance += config.radial_step) {
        const cv::Point2d samplePoint = circleCenter + unitDirection * distance;
        double intensity = 0.0;
        if (bilinearSample(gray64, samplePoint, intensity)) {
            profile.push_back({ distance, intensity });
        }
    }

    return !profile.empty();
}

bool super_edge_detector::cal_profile_gradient(const std::vector<RadialProfileSample> &profile, const detector_config &config, std::vector<double> &gradients, std::vector<double> &x_values)
{
    if (profile.size() < 3) {
        return false;
    }

    for (size_t i = 1; i < profile.size() - 1; i++)
    {
        double grad = (profile[i + 1].intensity - profile[i - 1].intensity) / (profile[i + 1].distance - profile[i - 1].distance);
        gradients.push_back(grad);
        x_values.push_back(profile[i].distance);
    }
    return true;
}

bool super_edge_detector::ceres_optimization(const std::vector<double> &gradients, const std::vector<double> &x_values, const detector_config &config, double &optimized_value)
{
    double a = 127.5;
    double mu = x_values.size() / 2.0;
    double sigma1 = 2.0;
    double sigma2 = 2.0;

    ceres::Problem problem;
    for (size_t i = 0; i < x_values.size(); i++)
    {
        ceres::CostFunction* cost_function = new ceres::AutoDiffCostFunction<AsymmetricGaussianFunctor, 1, 1, 1, 1, 1>(
            new AsymmetricGaussianFunctor(x_values[i], gradients[i]));
        problem.AddResidualBlock(cost_function, nullptr, &a, &mu, &sigma1, &sigma2);
    }
    
    problem.SetParameterLowerBound(&sigma1, 0, 0.01);
    problem.SetParameterLowerBound(&sigma2, 0, 0.01);

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    if (summary.IsSolutionUsable()) {
        optimized_value = mu;
        //std::cout << "Optimization successful. Optimized value: " << optimized_value << std::endl;
    }
    
    return true;
}

void super_edge_detector::draw_subpixel_edges(cv::Mat& image, const std::vector<cv::Point2d>& edge_points, bool draw_lines, int shift)
{
    if (edge_points.empty() || image.empty()) {
        return;
    }

    // 计算放大倍数
    const double multiplier = static_cast<double>(1 << shift);
    std::vector<cv::Point> scaled_points;
    scaled_points.reserve(edge_points.size());

    // 计算放大后的绘制半径
    int radius = static_cast<int>(std::round(1.5 * multiplier));

    for (const auto& pt : edge_points)
    {
        // 转换为包含 shift 放大的整型坐标
        cv::Point scaled_pt(
            static_cast<int>(std::round(pt.x * multiplier)),
            static_cast<int>(std::round(pt.y * multiplier))
        );
        scaled_points.push_back(scaled_pt);

        // 绘制单个亚像素点 (红色)
        cv::circle(image, scaled_pt, radius, cv::Scalar(0, 0, 255), -1, cv::LINE_AA, shift);
    }

    // 绘制连线 (绿色多边形)
    if (draw_lines && scaled_points.size() >= 2)
    {
        std::vector<std::vector<cv::Point>> pts = { scaled_points };
        // true 表示闭合多边形
        cv::polylines(image, pts, true, cv::Scalar(0, 255, 0), 1, cv::LINE_AA, shift);
    }
}
