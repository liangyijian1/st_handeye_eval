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

inline double cubicWeight(double x) {
    const double a = -0.75;  // opencv default cubic interpolation parameter
    x = std::abs(x);
    if (x <= 1.0) {
        return (a + 2.0) * x * x * x - (a + 3.0) * x * x + 1.0;
    } else if (x <= 2.0) {
        return a * x * x * x - 5.0 * a * x * x + 8.0 * a * x - 4.0 * a;
    }
    return 0.0;
}

bool bicubicSample(const cv::Mat& gray64, const cv::Point2d& point, double& value)
{
    if (point.x < 0.0 || point.y < 0.0 ||
        point.x > static_cast<double>(gray64.cols - 1) ||
        point.y > static_cast<double>(gray64.rows - 1)) {
        return false;
    }

    const int x0 = static_cast<int>(std::floor(point.x));
    const int y0 = static_cast<int>(std::floor(point.y));
    const double dx = point.x - static_cast<double>(x0);
    const double dy = point.y - static_cast<double>(y0);

    double weights_x[4] = {
        cubicWeight(1.0 + dx),
        cubicWeight(dx),
        cubicWeight(1.0 - dx),
        cubicWeight(2.0 - dx)
    };

    double weights_y[4] = {
        cubicWeight(1.0 + dy),
        cubicWeight(dy),
        cubicWeight(1.0 - dy),
        cubicWeight(2.0 - dy)
    };

    double result = 0.0;

    for (int j = -1; j <= 2; ++j) {
        int py = std::max(0, std::min(y0 + j, gray64.rows - 1));
        double row_val = 0.0;
        
        for (int i = -1; i <= 2; ++i) {
            int px = std::max(0, std::min(x0 + i, gray64.cols - 1));
            row_val += gray64.at<double>(py, px) * weights_x[i + 1];
        }
        result += row_val * weights_y[j + 1];
    }

    if (std::isfinite(result)) {
        value = result;
        return true;
    }
    
    return false;
}

bool plot_profile(
    const std::vector<double>& x_values, 
    const std::vector<double>& gradients, 
    const std::string& save_path
)
{
    if (x_values.empty() || gradients.empty() || save_path.empty()) {
        return false;
    }

    int width = 800;
    int height = 600;
    int margin = 60;
    cv::Mat plot(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    double min_x = x_values.front();
    double max_x = x_values.back();
    double max_y = *std::max_element(gradients.begin(), gradients.end());
    max_y = std::max(max_y * 1.2, 0.1);

    auto map_x = [&](double x) {
        return margin + static_cast<int>((x - min_x) / (max_x - min_x) * (width - 2 * margin));
    };
    auto map_y = [&](double y) {
        return height - margin - static_cast<int>((y / max_y) * (height - 2 * margin));
    };

    cv::line(plot, cv::Point(margin, height - margin), cv::Point(width - margin, height - margin), cv::Scalar(0, 0, 0), 2); // X轴
    cv::line(plot, cv::Point(margin, height - margin), cv::Point(margin, margin), cv::Scalar(0, 0, 0), 2); // Y轴

    for (size_t i = 0; i < x_values.size(); ++i) {
        cv::Point pt(map_x(x_values[i]), map_y(gradients[i]));
        cv::circle(plot, pt, 4, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
    }

    // 保存图像
    cv::imwrite(save_path, plot);
    return true;
}

bool super_edge_detector::detect_edges()
{
    // 1. Load images from root_path_
    auto root_path = fs::path(root_path_);
    fs::path out_root = root_path / "output";
    fs::path rough_dir = out_root / "rough_detection";
    fs::path fine_dir = out_root / "fine_detection";
    fs::path crop_dir = out_root / "fine_subpixel_crops";
    fs::path summary_dir = out_root / "summary";

    fs::create_directories(rough_dir);
    fs::create_directories(fine_dir);
    fs::create_directories(crop_dir);
    fs::create_directories(summary_dir);
    for (const auto& entry : fs::directory_iterator(root_path))
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
                std::string img_name = path.stem().string();
                std::string rough_save_path = (rough_dir / (img_name + "_rough.jpg")).string();
                detect_circles(image, circles, rough_save_path);
                std::cout << "Detected " << circles.size() << " circles in image: " << path << std::endl;
                cv::Mat displayImage = image.clone();
                cv::Mat profileImage;
                fs::path img_crop_dir = crop_dir / img_name;
                fs::create_directories(img_crop_dir);
                fs::path img_plot_dir = crop_dir / img_name / "plots";
                fs::create_directories(img_plot_dir);

                if (!makeGray64F(image, profileImage)) {
                    std::cerr << "Failed to build radial profile image: " << path << std::endl;
                    continue;
                }
                std::ofstream summary_file((summary_dir / (img_name + "_summary.txt")).string());
                summary_file << "file: " << path.filename().string() << "\n";
                summary_file << "blob circle count: " << circles.size() << "\n\n";
                int fine_detected_circles_count = 0;
                for (size_t i = 0; i < circles.size(); i++)
                {
                    std::vector<cv::Point2d> edgePoints;
                    double original_radius = static_cast<double>(circles[i][2]);
                    double total_optimization_shift = 0.0;
                    int valid_points = 0;
                    std::vector<int> rayIndices; // successful points ray indices
                    std::vector<std::vector<cv::Point2d>> samplePoints2D;
                    
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
                        std::string plot_path = (img_plot_dir / ("circle_" + std::to_string(i) + "_ray_" + std::to_string(j) + "_curve.jpg")).string();
                        AsymmetricGaussianResult result;
                        ceres::Solver::Summary summary;
                        if (ceres_optimization(gradients, x_values, config, result, summary)){
                            // 6. map edge position back to image
                            edgePoints.emplace_back(circles[i][0] + result.mu * direction.x, circles[i][1] + result.mu * direction.y);
                            total_optimization_shift += std::abs(result.mu - original_radius);
                            valid_points++;
                            rayIndices.push_back(j);

                            std::vector<cv::Point2d> current_ray_samples;
                            for (const auto& s : profile) {
                                current_ray_samples.emplace_back(circles[i][0] + s.distance * direction.x, circles[i][1] + s.distance * direction.y);
                            }
                            samplePoints2D.push_back(current_ray_samples);
                            
                        }
                        plot_fitting_curve(x_values, gradients, result.a, result.mu, result.sigma1, result.sigma2, summary, plot_path, j);
                    }
                    if (valid_points > 0)
                    {
                        fine_detected_circles_count++;
                        double avg_shift = total_optimization_shift / valid_points;
                        
                        summary_file << "The " << i + 1 << " circle (Center X: " << circles[i][0] << ", Y: " << circles[i][1] << ")\n"
                                     << "   Sub-pixel edge points generated: " << valid_points << "/" << config.num_directions << "\n"
                                     << "   Average coordinate correction (relative to original coarse radius): " << std::fixed << std::setprecision(4) << avg_shift << " px\n\n";

                        int margin = static_cast<int>(config.radial_margin) + 15;
                        int cx = cvRound(circles[i][0]);
                        int cy = cvRound(circles[i][1]);
                        int r = cvRound(original_radius);

                        cv::Rect roi(cx - r - margin, cy - r - margin, 2 * (r + margin), 2 * (r + margin));
                        cv::Rect img_rect(0, 0, image.cols, image.rows);
                        roi = roi & img_rect; // Intersect to prevent boundary overflow

                        if (roi.width > 0 && roi.height > 0) {
                            cv::Mat crop = image(roi).clone();
                            double scale = 8.0; // 8× magnification
                            cv::Mat high_res_crop;
                            cv::resize(crop, high_res_crop, cv::Size(), scale, scale, cv::INTER_CUBIC);

                            std::vector<cv::Point2d> local_edges;
                            std::vector<std::vector<cv::Point2d>> local_sample_points;

                            for (size_t k = 0; k < edgePoints.size(); k++) {
                                local_edges.emplace_back((edgePoints[k].x - roi.x) * scale, (edgePoints[k].y - roi.y) * scale);
                                
                                std::vector<cv::Point2d> mapped_ray;
                                for (const auto& pt : samplePoints2D[k]) {
                                    mapped_ray.emplace_back((pt.x - roi.x) * scale, (pt.y - roi.y) * scale);
                                }
                                local_sample_points.push_back(mapped_ray);
                            }
                            // 7. draw detected edges on the image and save
                            draw_subpixel_edges(high_res_crop, local_edges, rayIndices, local_sample_points, true, 8);
                            
                            std::string crop_save_path = (img_crop_dir / ("circle_" + std::to_string(i) + ".jpg")).string();
                            cv::imwrite(crop_save_path, high_res_crop);
                        }
                    }
                    draw_subpixel_edges(displayImage, edgePoints, rayIndices, samplePoints2D, true, 1);
                }
                summary_file << "========================\n";
                summary_file << "Fine localization successful circle count: " << fine_detected_circles_count << "\n";
                summary_file.close();
                std::string fine_save_path = (fine_dir / (img_name + "_fine.jpg")).string();
                cv::imwrite(fine_save_path, displayImage);
            }
        }
    }
    return true;
}

bool super_edge_detector::detect_circles(const cv::Mat& image, std::vector<cv::Vec3f>& circles, const std::string& save_path)
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
    cv::imwrite(save_path, display);
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
        if (bicubicSample(gray64, samplePoint, intensity)) {
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
        gradients.push_back(std::abs(grad));
        x_values.push_back(profile[i].distance);
    }
    return true;
}

bool super_edge_detector::ceres_optimization(
    const std::vector<double> &gradients, 
    const std::vector<double> &x_values, 
    const detector_config &config, 
    AsymmetricGaussianResult& result,
    ceres::Solver::Summary& summary
)
{
    auto max_it = std::max_element(gradients.begin(), gradients.end());
    auto max_idx = std::distance(gradients.begin(), max_it);
    if (*max_it < 5.0) return false;

    double a = *max_it;
    double mu = x_values[max_idx];
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
    //problem.SetParameterLowerBound(&a, 0, 0.01); // Amplitude is always positive
    problem.SetParameterLowerBound(&mu, 0, x_values.front());
    problem.SetParameterUpperBound(&mu, 0, x_values.back());

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 100;

    ceres::Solve(options, &problem, &summary);

    result.a = a;
    result.mu = mu;
    result.sigma1 = sigma1;
    result.sigma2 = sigma2;

    if (summary.final_cost > 3000.0 || (sigma1 + sigma2) > 2.0)
    {
        return false;
    }
    
    if (!summary.IsSolutionUsable()) {
        return false;
    }

    return true;
}

void super_edge_detector::draw_subpixel_edges(
    cv::Mat& image, 
    const std::vector<cv::Point2d>& edge_points, 
    const std::vector<int>& ray_indices, 
    const std::vector<std::vector<cv::Point2d>>& sample_points,
    bool draw_circles, int shift)
{
    if (edge_points.empty() || image.empty()) {
        return;
    }

    const double multiplier = static_cast<double>(1 << shift);
    std::vector<cv::Point> scaled_points;
    scaled_points.reserve(edge_points.size());

    int radius = static_cast<int>(std::round(1.5 * multiplier));

    if (!sample_points.empty() && sample_points.size() == edge_points.size()) {
        // 采样点画小一点，半径大致是红点的一半以内
        int sample_radius = std::max(1, static_cast<int>(std::round(0.6 * multiplier))); 
        for (size_t i = 0; i < sample_points.size(); ++i) {
            for (const auto& pt : sample_points[i]) {
                cv::Point scaled_pt(
                    static_cast<int>(std::round(pt.x * multiplier)),
                    static_cast<int>(std::round(pt.y * multiplier))
                );
                cv::circle(image, scaled_pt, sample_radius, cv::Scalar(0, 255, 255), -1, cv::LINE_AA, shift);
            }
        }
    }

    for (size_t i = 0; i < edge_points.size(); ++i)
    {
        const auto& pt = edge_points[i];
        cv::Point scaled_pt(
            static_cast<int>(std::round(pt.x * multiplier)),
            static_cast<int>(std::round(pt.y * multiplier))
        );
        scaled_points.push_back(scaled_pt);

        cv::circle(image, scaled_pt, radius, cv::Scalar(0, 0, 255), -1, cv::LINE_AA, shift);

        if (!ray_indices.empty() && ray_indices.size() == edge_points.size()) {
            cv::Point text_pt(static_cast<int>(std::round(pt.x)), static_cast<int>(std::round(pt.y)));
            text_pt.x += 3;
            text_pt.y += 3; 
            cv::putText(image, std::to_string(ray_indices[i]), text_pt, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        }
    }

    // fit an ellipse if there are enough points
    if (draw_circles && edge_points.size() >= 5)
    {
        cv::RotatedRect ellipse = cv::fitEllipse(scaled_points);
        auto center = ellipse.center;
        auto axes = ellipse.size;
        auto angle = ellipse.angle;
        cv::ellipse(image, center, cv::Size(axes.width / 2.0, axes.height / 2.0), angle, 0, 360, cv::Scalar(0, 255, 0), 1, cv::LINE_AA, shift);
    }
}


void super_edge_detector::plot_fitting_curve(
    const std::vector<double>& x_values, 
    const std::vector<double>& gradients, 
    double a, double mu, double sigma1, double sigma2, 
    const ceres::Solver::Summary& summary,
    const std::string& save_path,
    int ray_index
)
{
    if (x_values.empty() || gradients.empty() || save_path.empty()) {
        return;
    }

    int width = 800;
    int height = 600;
    int margin = 60;
    cv::Mat plot(height, width, CV_8UC3, cv::Scalar(255, 255, 255));

    double min_x = x_values.front();
    double max_x = x_values.back();
    double max_y = *std::max_element(gradients.begin(), gradients.end());
    max_y = std::max(max_y * 1.2, 0.1);

    // 映射函数：将数据坐标转化为图像像素坐标
    auto map_x = [&](double x) {
        return margin + static_cast<int>((x - min_x) / (max_x - min_x) * (width - 2 * margin));
    };
    auto map_y = [&](double y) {
        return height - margin - static_cast<int>((y / max_y) * (height - 2 * margin));
    };

    cv::line(plot, cv::Point(margin, height - margin), cv::Point(width - margin, height - margin), cv::Scalar(0, 0, 0), 2); // X轴
    cv::line(plot, cv::Point(margin, height - margin), cv::Point(margin, margin), cv::Scalar(0, 0, 0), 2); // Y轴
    auto final_cost = summary.final_cost;

    std::string title = "Final Cost: " + std::to_string(final_cost) + ", Sigma1: " + std::to_string(sigma1) + ", Sigma2: " + std::to_string(sigma2);
    cv::putText(plot, title, cv::Point(margin, margin - 20), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 0), 1);

    for (size_t i = 0; i < x_values.size(); ++i) {
        cv::Point pt(map_x(x_values[i]), map_y(gradients[i]));
        cv::circle(plot, pt, 4, cv::Scalar(255, 0, 0), -1, cv::LINE_AA);
    }

    int num_pts = 200;
    cv::Point prev_pt(-1, -1);
    for (int i = 0; i <= num_pts; ++i) {
        double x = min_x + i * (max_x - min_x) / num_pts;
        
        // Asymmetric Gaussian
        double sig1 = std::abs(sigma1) + 1e-6;
        double sig2 = std::abs(sigma2) + 1e-6;
        double diff = x - mu;
        double current_sigma = (diff < 0.0) ? sig1 : sig2;
        double constant_term = 2.0 / std::sqrt(2.0 * CV_PI);
        double exp_term = std::exp(-(diff * diff) / (2.0 * current_sigma * current_sigma));
        double y = a * constant_term * (1.0 / (sig1 + sig2)) * exp_term;

        cv::Point pt(map_x(x), map_y(y));
        if (prev_pt.x != -1) {
            cv::line(plot, prev_pt, pt, cv::Scalar(0, 0, 255), 2, cv::LINE_AA);
        }
        prev_pt = pt;
    }

    int mu_x = map_x(mu);
    if (mu_x >= margin && mu_x <= width - margin) {
        cv::line(plot, cv::Point(mu_x, height - margin), cv::Point(mu_x, margin), cv::Scalar(0, 200, 0), 1, cv::LINE_AA);
        cv::putText(plot, "Edge(Mu)", cv::Point(mu_x + 5, margin + 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 200, 0), 1);
    }

    // 保存图像
    cv::imwrite(save_path, plot);
}

