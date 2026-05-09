#include "super_edge_detector.h"

#include <algorithm>
#include <cmath>
#include <iostream>


static bool makeGray64F(const cv::Mat& image, cv::Mat& gray64)
{
    if (image.empty()) return false;

    cv::Mat gray;
    switch (image.channels()) {
        case 1: gray = image; break;
        case 3: cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY); break;
        case 4: cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY); break;
        default: return false;
    }

    if (gray.depth() == CV_64F)
        gray64 = gray;
    else
        gray.convertTo(gray64, CV_64F);
    return true;
}

double elegant_score(double x, double theta, double k = 3.0) {
    if (x <= 0) return 1.0;

    return 1.0 / (1.0 + std::pow(x / theta, k));
}

float calculate_right_fit_score(
    const std::vector<double>& x_values, 
    const std::vector<double>& gradients, 
    const AsymmetricGaussianResult& res)
{
    double sum_sq_err = 0.0;
    int count = 0;

    double sig_sum = std::abs(res.sigma1) + std::abs(res.sigma2) + 1e-6;
    double constant_term = res.a * (2.0 / std::sqrt(2.0 * CV_PI)) * (1.0 / sig_sum);
    double two_sig2_sq = 2.0 * res.sigma2 * res.sigma2 + 1e-9;

    for (size_t i = 0; i < x_values.size(); ++i) {
        if (x_values[i] < res.mu) continue;

        double diff = x_values[i] - res.mu;
        double y_pred = constant_term * std::exp(-(diff * diff) / two_sig2_sq);
        
        double err = gradients[i] - y_pred;
        sum_sq_err += err * err;
        count++;
    }

    if (count == 0) return 0.0;

    double rmse = std::sqrt(sum_sq_err / count);

    return static_cast<float>(rmse);
}

static void draw_profile_plot(const std::vector<RadialProfileSample>& prof, const std::string& title, const std::string& save_path, double mu, double distance_scale) {
    if (prof.empty()) return;

    int pw = 800, ph = 600, pm = 60;
    cv::Mat prof_plot(ph, pw, CV_8UC3, cv::Scalar(255, 255, 255));

    double px_min = prof.front().distance / distance_scale;
    double px_max = prof.back().distance / distance_scale;

    auto [min_it, max_it] = std::minmax_element(prof.begin(), prof.end(),
        [](const RadialProfileSample& a, const RadialProfileSample& b) {
            return a.intensity < b.intensity;
        });
    double py_min = min_it->intensity;
    double py_max = max_it->intensity;

    if (py_max - py_min < 1e-6) py_max = py_min + 1.0;
    double py_range = py_max - py_min;
    py_min -= py_range * 0.05;
    py_max += py_range * 0.05;

    auto map_x = [&](double x) { return pm + static_cast<int>((x - px_min) / (px_max - px_min) * (pw - 2 * pm)); };
    auto map_y = [&](double y) { return ph - pm - static_cast<int>((y - py_min) / (py_max - py_min) * (ph - 2 * pm)); };

    cv::line(prof_plot, cv::Point(pm, ph - pm), cv::Point(pw - pm, ph - pm), cv::Scalar(0, 0, 0), 2);
    cv::line(prof_plot, cv::Point(pm, ph - pm), cv::Point(pm, pm), cv::Scalar(0, 0, 0), 2);
    cv::putText(prof_plot, title, cv::Point(pm, pm - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    cv::putText(prof_plot, "Distance", cv::Point(pw / 2 - 30, ph - 10), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);
    cv::putText(prof_plot, "Intensity", cv::Point(5, pm - 5), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 0, 0), 1);

    cv::Point prev_p(-1, -1);
    for (const auto& s : prof) {
        cv::Point p(map_x(s.distance / distance_scale), map_y(s.intensity));
        cv::circle(prof_plot, p, 3, cv::Scalar(200, 0, 0), -1, cv::LINE_AA);
        if (prev_p.x != -1) cv::line(prof_plot, prev_p, p, cv::Scalar(200, 120, 0), 1, cv::LINE_AA);
        prev_p = p;
    }

    int mu_px = map_x(mu);
    if (mu_px >= pm && mu_px <= pw - pm) {
        cv::line(prof_plot, cv::Point(mu_px, ph - pm), cv::Point(mu_px, pm), cv::Scalar(0, 200, 0), 1, cv::LINE_AA);
        cv::putText(prof_plot, "Edge(Mu)", cv::Point(mu_px + 5, pm + 20), cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(0, 200, 0), 1);
    }
    cv::imwrite(save_path, prof_plot);
}

void super_edge_detector::radial_profile_manual(
    const cv::Mat& gray64, 
    const cv::Vec3f& center, 
    const cv::Point2d& unitDirection, 
    double margin, 
    double step, 
    std::vector<RadialProfileSample>& profile)
{
    const double radius = static_cast<double>(center[2]);
    const cv::Point2d circleCenter(static_cast<double>(center[0]), static_cast<double>(center[1]));

    const double startDistance = std::max(0.0, radius - margin);
    const double endDistance = radius + margin;

    const int num_samples = static_cast<int>(std::floor((endDistance - startDistance) / step)) + 1;
    if (num_samples <= 0) return;

    cv::Mat map_x(1, num_samples, CV_32FC1);
    cv::Mat map_y(1, num_samples, CV_32FC1);
    float* p_map_x = map_x.ptr<float>(0);
    float* p_map_y = map_y.ptr<float>(0);
    std::vector<double> distances(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        double distance = startDistance + i * step;
        distances[i] = distance;
        p_map_x[i] = static_cast<float>(circleCenter.x + unitDirection.x * distance);
        p_map_y[i] = static_cast<float>(circleCenter.y + unitDirection.y * distance);
    }

    cv::Mat profile_result;
    cv::remap(gray64, profile_result, map_x, map_y, cv::INTER_CUBIC, cv::BORDER_REPLICATE);

    const double* p_res = profile_result.ptr<double>(0);
    profile.resize(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        profile[i] = { distances[i], p_res[i] };
    }
}

void super_edge_detector::cal_profile_gradient_manual(
    const std::vector<RadialProfileSample>& profile, 
    double step, 
    int polarity,
    std::vector<double>& gradients, 
    std::vector<double>& x_values)
{
    if (profile.size() < 5) return;

    const double inv_denominator = 1.0 / (12.0 * step);
    const size_t n = profile.size();

    for (size_t i = 2; i < n - 2; ++i) {
        double grad = (profile[i - 2].intensity * 1.0 + 
                       profile[i - 1].intensity * -8.0 + 
                       profile[i + 1].intensity * 8.0 + 
                       profile[i + 2].intensity * -1.0) * inv_denominator;

        double final_grad = 0.0;
        if (polarity == 1)      final_grad = std::max(0.0, grad);
        else if (polarity == -1) final_grad = std::max(0.0, -grad);
        else                     final_grad = std::abs(grad);

        gradients.push_back(final_grad);
        x_values.push_back(profile[i].distance);
    }
}

bool super_edge_detector::detect_edges()
{
    const auto root_path = fs::path(root_path_);
    const fs::path out_root = root_path / "output";
    const fs::path rough_dir = out_root / "rough_detection";
    const fs::path fine_dir = out_root / "fine_detection";
    const fs::path crop_dir = out_root / "fine_subpixel_crops";
    const fs::path summary_dir = out_root / "summary";

    fs::create_directories(rough_dir);
    fs::create_directories(fine_dir);
    fs::create_directories(crop_dir);
    fs::create_directories(summary_dir);

    // Pre-compute direction vectors
    const int nd = config.num_directions;
    std::vector<cv::Point2d> directions(nd);
    for (int j = 0; j < nd; ++j) {
        double angle = 2.0 * CV_PI * j / nd;
        directions[j] = { std::cos(angle), std::sin(angle) };
    }

    for (const auto& entry : fs::directory_iterator(root_path))
    {
        if (!entry.is_regular_file() || entry.path().extension() != ".bmp")
            continue;

        const auto& path = entry.path();
        cv::Mat image = cv::imread(path.string());
        if (image.empty()) {
            std::cerr << "Failed to load image: " << path << std::endl;
            continue;
        }

        // 1. Rough circle detection
        std::vector<cv::Vec3f> circles;
        std::string img_name = path.stem().string();
        detect_circles(image, circles, (rough_dir / (img_name + "_rough.jpg")).string());
        std::cout << "Detected " << circles.size() << " circles in image: " << path << std::endl;

        // 2. Convert to gray64 once for the whole image (avoid per-ray conversion)
        cv::Mat gray64;
        if (!makeGray64F(image, gray64)) {
            std::cerr << "Failed to build radial profile image: " << path << std::endl;
            continue;
        }

        // Setup output directories (only if saving crops/plots)
        fs::path img_crop_dir, img_plot_dir;
        if (config.save_crops || config.save_plots) {
            img_crop_dir = crop_dir / img_name;
            fs::create_directories(img_crop_dir);
            if (config.save_plots) {
                img_plot_dir = img_crop_dir / "plots";
                fs::create_directories(img_plot_dir);
            }
        }

        cv::Mat displayImage = image.clone();
        std::ofstream summary_file((summary_dir / (img_name + "_summary.txt")).string());
        summary_file << "file: " << path.filename().string() << "\n";
        summary_file << "detect circle count: " << circles.size() << "\n\n";
        int fine_detected_circles_count = 0;

        for (int i = 0; i < static_cast<int>(circles.size()); ++i)
        {
            const double cx_f = circles[i][0], cy_f = circles[i][1];
            const double original_radius = static_cast<double>(circles[i][2]);
            double total_optimization_shift = 0.0;
            int valid_points = 0;

            std::vector<cv::Point2d> edgePoints;
            std::vector<int> rayIndices;
            std::vector<std::vector<cv::Point2d>> samplePoints2D;
            edgePoints.reserve(nd);
            rayIndices.reserve(nd);
            samplePoints2D.reserve(nd);

            std::vector<AsymmetricGaussianResult> allResults(nd);
            std::vector<ceres::Solver::Summary> allSummaries(nd);
            std::vector<float> right_fit_scores;

            for (int j = 0; j < nd; ++j)
            {
                const auto& dir = directions[j];

                // 3. Radial profile
                std::vector<RadialProfileSample> profile;
                radial_profile(gray64, circles[i], dir, profile);

                // 4. Profile gradient
                std::vector<double> gradients, x_values;
                gradients.reserve(profile.size());
                x_values.reserve(profile.size());
                cal_profile_gradient(profile, gradients, x_values);

                // 5. Ceres optimization
                AsymmetricGaussianResult result;
                ceres::Solver::Summary summary;
                if (ceres_optimization(gradients, x_values, result, summary)) {
                    edgePoints.emplace_back(cx_f + result.mu * dir.x, cy_f + result.mu * dir.y);
                    total_optimization_shift += std::abs(result.mu - original_radius);
                    valid_points++;
                    rayIndices.push_back(j);

                    // Calculate right fit score
                    right_fit_scores.push_back(calculate_right_fit_score(x_values, gradients, result));

                    if (config.save_crops) {
                        std::vector<cv::Point2d> ray_samples;
                        ray_samples.reserve(profile.size());
                        for (const auto& s : profile)
                            ray_samples.emplace_back(cx_f + s.distance * dir.x, cy_f + s.distance * dir.y);
                        samplePoints2D.push_back(std::move(ray_samples));
                    }

                    if (config.save_plots) {
                    fs::path plot_path_fs = img_plot_dir / ("circle_" + std::to_string(i) + "_ray_" + std::to_string(j) + "_curve.jpg");
                    std::string final_save_path = plot_path_fs.generic_string();
                    plot_fitting_curve(x_values, gradients, result.a, result.mu, result.sigma1, result.sigma2, summary, final_save_path, j);

                    std::string prof_save_path = (img_plot_dir / ("circle_" + std::to_string(i) + "_ray_" + std::to_string(j) + "_profile.jpg")).generic_string();
                    draw_profile_plot(profile, "Original Grayscale Profile | Circle " + std::to_string(i) + " Ray " + std::to_string(j), prof_save_path, result.mu, 1.0);

                    if (config.save_crops && !rayIndices.empty()) {
                        int pad = static_cast<int>(config.radial_margin) + 15;
                        int cx = cvRound(cx_f), cy = cvRound(cy_f), r = cvRound(original_radius);
                        cv::Rect roi(cx - r - pad, cy - r - pad, 2 * (r + pad), 2 * (r + pad));
                        roi &= cv::Rect(0, 0, image.cols, image.rows);
                        constexpr double scale = 8.0;
                        cv::Mat high_res_crop;
                        cv::resize(image(roi), high_res_crop, cv::Size(), scale, scale, cv::INTER_CUBIC);
                        cv::Mat gray_high_res_crop;
                        makeGray64F(high_res_crop, gray_high_res_crop);

                        cv::Point2d local_center((cx_f - roi.x) * scale, (cy_f - roi.y) * scale);
                        double local_radius = original_radius * scale;
                        cv::Vec3f debug_circle(static_cast<float>(local_center.x), static_cast<float>(local_center.y), static_cast<float>(local_radius));
                        
                        double debug_margin = config.radial_margin * scale;
                        double debug_step = config.radial_step * scale; 

                        std::vector<RadialProfileSample> profile_debug;
                        radial_profile_manual(gray_high_res_crop, debug_circle, directions[j], 
                                            debug_margin, debug_step, profile_debug);

                        std::vector<double> gradients_debug, x_values_debug;
                        cal_profile_gradient_manual(profile_debug, debug_step, config.edge_polarity, 
                                                gradients_debug, x_values_debug);

                        for (auto& x : x_values_debug) {
                            x = x / scale;
                        }

                        std::string debug_plot_path = (img_plot_dir / ("circle_" + std::to_string(i) + "_ray_" + std::to_string(j) + "_debug_curve.jpg")).generic_string();
                        plot_fitting_curve(x_values_debug, gradients_debug, 0, 0, 0, 0,
                                        ceres::Solver::Summary(), debug_plot_path, j);

                        std::string debug_prof_save_path = (img_plot_dir / ("circle_" + std::to_string(i) + "_ray_" + std::to_string(j) + "_debug_profile.jpg")).generic_string();
                        draw_profile_plot(profile_debug, "Upscaled Grayscale Profile | Circle " + std::to_string(i) + " Ray " + std::to_string(j), debug_prof_save_path, result.mu, scale);
                    }
                }
                }
                allResults[j] = result;
                allSummaries[j] = summary;
            }

            if (valid_points > 5)
            {
                fine_detected_circles_count++;
                double avg_shift = total_optimization_shift / valid_points;

                // output confidence
                double confidence = 0.0;
                double S_cover = static_cast<double>(valid_points) / nd;

                float sigma_sum = 0.0f;
                for (int j : rayIndices) sigma_sum += static_cast<float>(allResults[j].sigma1 + allResults[j].sigma2);
                float sigma_avg = sigma_sum / valid_points;
                double S_sharpness = elegant_score(sigma_avg, config.ceres_sigma_th, 4.0);

                float cost_sum = 0.0f;
                for (int j : rayIndices) cost_sum += static_cast<float>(allSummaries[j].final_cost);
                float cost_avg = cost_sum / valid_points;
                double S_cost = elegant_score(cost_avg, config.ceres_cost_th, 6.0);

                const int K = 6;
                std::vector<int> sector_hit(K, 0);
                for (int j : rayIndices) {
                    double angle = 2.0 * CV_PI * j / nd;
                    int sector = static_cast<int>(angle / (2.0 * CV_PI) * K) % K;
                    sector_hit[sector] = 1;
                }
                int occupied = std::accumulate(sector_hit.begin(), sector_hit.end(), 0);
                double S_angular = static_cast<double>(occupied) / K;

                confidence = config.summary_weights.coverage_weight * S_cover + 
                             config.summary_weights.sharpness_weight * S_sharpness + 
                             config.summary_weights.cost_weight * S_cost + 
                             config.summary_weights.angular_distribution_weight * S_angular;
                

                summary_file << "The " << i << " circle (Center X: " << cx_f << ", Y: " << cy_f << ")\n"
                             << "   Sub-pixel edge points generated: " << valid_points << "/" << nd << "\n"
                             << "   Average coordinate correction: " << std::fixed << std::setprecision(4) << avg_shift << " px\n"
                             << "   Confidence details after weighting:\n"
                             << "    - Coverage: " << std::fixed << std::setprecision(4) << config.summary_weights.coverage_weight * S_cover << "\n"
                             << "    - Sharpness: " << std::fixed << std::setprecision(4) << config.summary_weights.sharpness_weight * S_sharpness << "\n"
                             << "    - Cost: " << std::fixed << std::setprecision(4) << config.summary_weights.cost_weight * S_cost << "\n"
                             << "    - Angular distribution: " << std::fixed << std::setprecision(4) << config.summary_weights.angular_distribution_weight * S_angular << "\n"
                             << "    - Overall confidence: " << std::fixed << std::setprecision(4) << confidence << "\n\n";


                if (config.save_crops) {
                    int pad = static_cast<int>(config.radial_margin) + 15;
                    int cx = cvRound(cx_f), cy = cvRound(cy_f), r = cvRound(original_radius);
                    cv::Rect roi(cx - r - pad, cy - r - pad, 2 * (r + pad), 2 * (r + pad));
                    roi &= cv::Rect(0, 0, image.cols, image.rows);

                    if (roi.width > 0 && roi.height > 0) {
                        constexpr double scale = 8.0;
                        cv::Mat high_res_crop;
                        cv::resize(image(roi), high_res_crop, cv::Size(), scale, scale, cv::INTER_CUBIC);

                        std::vector<cv::Point2d> local_edges;
                        std::vector<std::vector<cv::Point2d>> local_samples;
                        local_edges.reserve(edgePoints.size());
                        local_samples.reserve(edgePoints.size());
                        for (size_t k = 0; k < edgePoints.size(); ++k) {
                            local_edges.emplace_back((edgePoints[k].x - roi.x) * scale, (edgePoints[k].y - roi.y) * scale);
                            std::vector<cv::Point2d> mapped;
                            mapped.reserve(samplePoints2D[k].size());
                            for (const auto& pt : samplePoints2D[k])
                                mapped.emplace_back((pt.x - roi.x) * scale, (pt.y - roi.y) * scale);
                            local_samples.push_back(std::move(mapped));
                        }

                        cv::imwrite((img_crop_dir / ("circle_" + std::to_string(i) + "_ori.jpg")).string(), high_res_crop);
                        draw_subpixel_edges(high_res_crop, local_edges, rayIndices, local_samples, 8, 
                            DrawMode::DRAW_EDGES | DrawMode::DRAW_CENTERS | DrawMode::DRAW_PROFILES | DrawMode::DRAW_SAMPLES);
                        cv::imwrite((img_crop_dir / ("circle_" + std::to_string(i) + ".jpg")).string(), high_res_crop);
                    }
                }

                draw_subpixel_edges(displayImage, edgePoints, rayIndices, samplePoints2D, 1, DrawMode::DRAW_PROFILES | DrawMode::DRAW_CENTERS);
            }
        }

        summary_file << "Fine localization successful circle count: " << fine_detected_circles_count << "\n";
        summary_file.close();
        cv::imwrite((fine_dir / (img_name + "_fine.jpg")).string(), displayImage);
    }
    return true;
}

bool super_edge_detector::detect_circles(const cv::Mat& image, std::vector<cv::Vec3f>& circles, const std::string& save_path)
{
    circles.clear();
    if (image.empty()) return false;

    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else if (image.channels() == 4)
        cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY);
    else if (image.channels() == 1)
        gray = image;
    else
        return false;

    if (gray.depth() != CV_8U) {
        cv::normalize(gray, gray, 0, 255, cv::NORM_MINMAX, CV_8U);
    }

    if (config.circle_detection_method == CIRCLE_DETECT_EDGE_DRAWING)
    {
        cv::Ptr<cv::ximgproc::EdgeDrawing> edge_detector = cv::ximgproc::createEdgeDrawing();
        edge_detector->params.EdgeDetectionOperator = cv::ximgproc::EdgeDrawing::LSD;
        edge_detector->params.GradientThresholdValue = 40;
        edge_detector->params.AnchorThresholdValue = 20;
        edge_detector->params.MinPathLength = 5;
        edge_detector->params.PFmode = false;
        edge_detector->params.NFAValidation = true;
        edge_detector->detectEdges(gray);

        std::vector<cv::Vec6d> ellipses;
        edge_detector->detectEllipses(ellipses);
        for (const auto& ellipse : ellipses) {
            const double radius = (ellipse[2] > 0.0)
                                      ? ellipse[2]
                                      : 0.5 * (std::abs(ellipse[3]) + std::abs(ellipse[4]));
            if (radius >= 4.0 && radius <= 15.0) {
                circles.emplace_back(static_cast<float>(ellipse[0]),
                                     static_cast<float>(ellipse[1]),
                                     static_cast<float>(radius));
            }
        }
    }
    else
    {
        cv::GaussianBlur(gray, gray, cv::Size(9, 9), 2, 2);
        cv::HoughCircles(gray, circles, cv::HOUGH_GRADIENT_ALT,
                         1.5,    // dp
                         30,     // minDist
                         180,    // param1
                         0.9,    // param2
                         4,      // minRadius
                         15);    // maxRadius
    }

    if (!save_path.empty()) {
        cv::Mat display = image.clone();
        for (const auto& c : circles) {
            cv::circle(display, cv::Point(cvRound(c[0]), cvRound(c[1])), cvRound(c[2]), cv::Scalar(0, 255, 0), 2);
            cv::circle(display, cv::Point(cvRound(c[0]), cvRound(c[1])), 3, cv::Scalar(0, 0, 255), -1);
        }
        cv::imwrite(save_path, display);
    }

    return true;
}

bool super_edge_detector::radial_profile(const cv::Mat& gray64, const cv::Vec3f& center, const cv::Point2d& direction, std::vector<RadialProfileSample>& profile)
{
    const double radius = static_cast<double>(center[2]);
    const double directionNorm = std::sqrt(direction.x * direction.x + direction.y * direction.y);
    if (!std::isfinite(radius) || radius <= 0.0 ||
        !std::isfinite(directionNorm) || directionNorm <= 0.0 ||
        config.radial_margin <= 0.0 || config.radial_step <= 0.0) {
        return false;
    }

    const cv::Point2d circleCenter(static_cast<double>(center[0]), static_cast<double>(center[1]));
    const cv::Point2d unitDirection(direction.x / directionNorm, direction.y / directionNorm);
    const double startDistance = std::max(0.0, radius - config.radial_margin);
    const double endDistance = radius + config.radial_margin;

    const int num_samples = static_cast<int>(std::floor((endDistance - startDistance) / config.radial_step)) + 1;
    if (num_samples <= 0) return false;

    cv::Mat map_x(1, num_samples, CV_32FC1);
    cv::Mat map_y(1, num_samples, CV_32FC1);
    float* p_map_x = map_x.ptr<float>(0);
    float* p_map_y = map_y.ptr<float>(0);
    std::vector<double> distances(num_samples);

    for (int i = 0; i < num_samples; ++i) {
        double distance = startDistance + i * config.radial_step;
        distances[i] = distance;
        p_map_x[i] = static_cast<float>(circleCenter.x + unitDirection.x * distance);
        p_map_y[i] = static_cast<float>(circleCenter.y + unitDirection.y * distance);
    }

    cv::Mat profile_result;
    cv::remap(gray64, profile_result, map_x, map_y, cv::INTER_CUBIC, cv::BORDER_REPLICATE);

    const double* p_res = profile_result.ptr<double>(0);
    profile.resize(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        profile[i] = { distances[i], p_res[i]};
    }
    return true;
}

bool super_edge_detector::cal_profile_gradient(const std::vector<RadialProfileSample>& profile, std::vector<double>& gradients, std::vector<double>& x_values)
{
    if (profile.size() < 5) return false;

    const double h = config.radial_step;
    // 5点求导权重系数：(-1, 8, 0, -8, 1) / (12 * h)
    const double inv_denominator = 1.0 / (12.0 * h);

    const size_t n = profile.size();
    gradients.reserve(n);
    x_values.reserve(n);

    for (size_t i = 2; i < n - 2; ++i)
    {
        double grad = (profile[i - 2].intensity * 1.0 + 
                       profile[i - 1].intensity * -8.0 + 
                       profile[i + 1].intensity * 8.0 + 
                       profile[i + 2].intensity * -1.0) * inv_denominator;

        double final_grad = 0.0;
        if (config.edge_polarity == 1) {
            final_grad = std::max(0.0, grad);
        } 
        else if (config.edge_polarity == -1) {
            final_grad = std::max(0.0, -grad);
        } 
        else {
            final_grad = std::abs(grad);
        }

        gradients.push_back(final_grad);
        x_values.push_back(profile[i].distance);
    }

    return !gradients.empty();
}

bool super_edge_detector::ceres_optimization(
    const std::vector<double>& gradients,
    const std::vector<double>& x_values,
    AsymmetricGaussianResult& result,
    ceres::Solver::Summary& summary)
{
    auto max_it = std::max_element(gradients.begin(), gradients.end());
    if (*max_it < 5.0) return false;

    auto max_idx = std::distance(gradients.begin(), max_it);
    double a = *max_it;
    double mu = x_values[max_idx];
    double sigma1 = 2.0;
    double sigma2 = 2.0;

    ceres::Problem problem;
    for (size_t i = 0; i < x_values.size(); ++i)
    {
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<AsymmetricGaussianFunctor, 1, 1, 1, 1, 1>(
                new AsymmetricGaussianFunctor(x_values[i], gradients[i])),
            nullptr, &a, &mu, &sigma1, &sigma2);
    }

    problem.SetParameterLowerBound(&sigma1, 0, 0.01);
    problem.SetParameterLowerBound(&sigma2, 0, 0.01);
    problem.SetParameterLowerBound(&mu, 0, x_values.front());
    problem.SetParameterUpperBound(&mu, 0, x_values.back());

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = false;
    options.max_num_iterations = 50;
    options.function_tolerance = 1e-5;
    options.gradient_tolerance = 1e-8;
    options.parameter_tolerance = 1e-6;

    ceres::Solve(options, &problem, &summary);

    result = { a, mu, sigma1, sigma2 };

    if (summary.final_cost > config.ceres_cost_th || (sigma1 + sigma2) > config.ceres_sigma_th)
        return false;

    return summary.IsSolutionUsable();
}

void super_edge_detector::draw_subpixel_edges(
    cv::Mat& image, 
    const std::vector<cv::Point2d>& edge_points, 
    const std::vector<int>& ray_indices, 
    const std::vector<std::vector<cv::Point2d>>& sample_points,
    int shift, DrawMode mode)
{
    if (edge_points.empty() || image.empty()) {
        return;
    }

    const double multiplier = static_cast<double>(1 << shift);
    std::vector<cv::Point> scaled_points;
    scaled_points.reserve(edge_points.size());

    int radius = static_cast<int>(std::round(1.5 * multiplier));

    // draw sample 
    if (!sample_points.empty() && sample_points.size() == edge_points.size() && (mode & DrawMode::DRAW_SAMPLES) != DrawMode::DRAW_NONE) {
        int sample_radius = std::max(1, static_cast<int>(std::round(0.6 * multiplier))); 
        for (int i = 0; i < sample_points.size(); ++i) {
            for (const auto& pt : sample_points[i]) {
                cv::Point scaled_pt(
                    static_cast<int>(std::round(pt.x * multiplier)),
                    static_cast<int>(std::round(pt.y * multiplier))
                );
                cv::circle(image, scaled_pt, sample_radius, cv::Scalar(0, 255, 255), -1, cv::LINE_AA, shift);
            }
        }
    }

    for (int i = 0; i < edge_points.size(); ++i)
    {
        const auto& pt = edge_points[i];
        cv::Point scaled_pt(
            static_cast<int>(std::round(pt.x * multiplier)),
            static_cast<int>(std::round(pt.y * multiplier))
        );
        scaled_points.push_back(scaled_pt);

        if (!ray_indices.empty() && ray_indices.size() == edge_points.size() && (mode & DrawMode::DRAW_EDGES) != DrawMode::DRAW_NONE) {
            cv::circle(image, scaled_pt, radius, cv::Scalar(0, 0, 255), -1, cv::LINE_AA, shift);
            cv::Point text_pt(static_cast<int>(std::round(pt.x)), static_cast<int>(std::round(pt.y)));
            text_pt.x += 3;
            text_pt.y += 3; 
            cv::putText(image, std::to_string(ray_indices[i]), text_pt, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        }
        
    } 

    // fit ellipse
    if (edge_points.size() >= 5)
    {
        cv::RotatedRect ellipse = cv::fitEllipse(scaled_points);
        auto center = ellipse.center;
        auto axes = ellipse.size;
        auto angle = ellipse.angle;
        if ((mode & DrawMode::DRAW_CENTERS) != DrawMode::DRAW_NONE) {
            cv::circle(image, center, radius, cv::Scalar(255, 0, 0), -1, cv::LINE_AA, shift);
        } 
        if ((mode & DrawMode::DRAW_PROFILES) != DrawMode::DRAW_NONE)
        {
            cv::ellipse(image, center, cv::Size(axes.width / 2.0, axes.height / 2.0), angle, 0, 360, cv::Scalar(0, 255, 0), 1, cv::LINE_AA, shift);
        }
        
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

    auto map_x = [&](double x) {
        return margin + static_cast<int>((x - min_x) / (max_x - min_x) * (width - 2 * margin));
    };
    auto map_y = [&](double y) {
        return height - margin - static_cast<int>((y / max_y) * (height - 2 * margin));
    };

    cv::line(plot, cv::Point(margin, height - margin), cv::Point(width - margin, height - margin), cv::Scalar(0, 0, 0), 2); // X轴
    cv::line(plot, cv::Point(margin, height - margin), cv::Point(margin, margin), cv::Scalar(0, 0, 0), 2); // Y轴
    auto final_cost = summary.final_cost;

    std::string cost_str = std::to_string(summary.final_cost);
    if (cost_str.length() > 6) cost_str = cost_str.substr(0, 6);
    std::string s1_str = std::to_string(sigma1);
    if (s1_str.length() > 4) s1_str = s1_str.substr(0, 4);
    std::string s2_str = std::to_string(sigma2);
    if (s2_str.length() > 4) s2_str = s2_str.substr(0, 4);

    std::string title = "Cost: " + cost_str + " | S1: " + s1_str + " | S2: " + s2_str;
    cv::putText(plot, title, cv::Point(margin, margin - 20), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);

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

    cv::imwrite(save_path, plot);
}

