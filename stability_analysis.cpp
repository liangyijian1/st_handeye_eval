#include "stability_analysis.h"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace {

double clamp01(double value)
{
    return std::clamp(value, 0.0, 1.0);
}

double score_small_is_good(double value, double threshold)
{
    if (threshold <= 0.0) {
        return 0.0;
    }
    return clamp01(1.0 - value / threshold);
}

double mean_value(const std::vector<double>& values)
{
    if (values.empty()) {
        return 0.0;
    }
    return std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(values.size());
}

double stddev_value(const std::vector<double>& values, double mean)
{
    if (values.empty()) {
        return 0.0;
    }

    double sum = 0.0;
    for (double value : values) {
        const double diff = value - mean;
        sum += diff * diff;
    }
    return std::sqrt(sum / static_cast<double>(values.size()));
}

cv::Point2d record_center(const CircleDetectionRecord& record)
{
    if (record.has_optimized_contour) {
        return cv::Point2d(record.optimized_contour.center.x, record.optimized_contour.center.y);
    }
    return cv::Point2d(record.rough_circle[0], record.rough_circle[1]);
}

double record_radius(const CircleDetectionRecord& record)
{
    if (record.has_optimized_contour) {
        const double major_axis = std::max(record.optimized_contour.size.width, record.optimized_contour.size.height);
        const double minor_axis = std::min(record.optimized_contour.size.width, record.optimized_contour.size.height);
        if (major_axis > 0.0 && minor_axis > 0.0) {
            return 0.25 * (major_axis + minor_axis);
        }
    }
    return static_cast<double>(record.rough_circle[2]);
}

double circular_mean_angle(const std::vector<double>& angles)
{
    if (angles.empty()) {
        return 0.0;
    }

    double sum_sin = 0.0;
    double sum_cos = 0.0;
    for (double angle : angles) {
        sum_sin += std::sin(angle);
        sum_cos += std::cos(angle);
    }
    return std::atan2(sum_sin, sum_cos);
}

double circular_distance(double a, double b)
{
    return std::atan2(std::sin(a - b), std::cos(a - b));
}

double circular_stddev(const std::vector<double>& angles)
{
    if (angles.empty()) {
        return 0.0;
    }

    const double mean_angle = circular_mean_angle(angles);
    double sum = 0.0;
    for (double angle : angles) {
        const double diff = circular_distance(angle, mean_angle);
        sum += diff * diff;
    }
    return std::sqrt(sum / static_cast<double>(angles.size()));
}

} // namespace

StabilityMetricResult evaluate_shape_stability(const CircleDetectionRecord& record, const detector_config& config,
    ShapeStabilityDetails& details)
{
    details.valid_ray_ratio = config.num_directions > 0
        ? static_cast<double>(record.valid_ray_count) / static_cast<double>(config.num_directions)
        : 0.0;

    std::vector<double> sigmas;
    std::vector<double> costs;
    for (const auto& ray : record.rays) {
        if (!ray.optimization_succeeded) {
            continue;
        }
        sigmas.push_back(ray.sigma1 + ray.sigma2);
        costs.push_back(ray.final_cost);
    }

    const double sigma_mean = mean_value(sigmas);
    const double sigma_stddev = stddev_value(sigmas, sigma_mean);
    const double cost_mean = mean_value(costs);

    details.sharpness_consistency = score_small_is_good(sigma_mean + sigma_stddev, config.ceres_sigma_th * 1.5);
    details.fit_cost_quality = score_small_is_good(cost_mean, config.ceres_cost_th * 1.2);
    details.angular_coverage = config.summary_weights.angular_distribution_weight > 0.0
        ? clamp01(record.angular_score / config.summary_weights.angular_distribution_weight)
        : 0.0;

    if (record.has_optimized_contour) {
        const double major_axis = std::max(record.optimized_contour.size.width, record.optimized_contour.size.height);
        const double minor_axis = std::min(record.optimized_contour.size.width, record.optimized_contour.size.height);
        details.contour_roundness = major_axis > 0.0 ? clamp01(minor_axis / major_axis) : 0.0;
    }

    StabilityMetricResult result;
    result.score = 0.25 * details.valid_ray_ratio
        + 0.20 * details.sharpness_consistency
        + 0.20 * details.fit_cost_quality
        + 0.15 * details.angular_coverage
        + 0.20 * details.contour_roundness;
    result.passed = result.score >= config.confidence_threshold;
    return result;
}

StabilityMetricResult evaluate_temporal_stability(const std::vector<CircleDetectionRecord>& track,
    TemporalStabilityDetails& details)
{
    std::vector<double> centers;
    std::vector<double> radii;
    std::vector<double> confidences;

    if (track.empty()) {
        return {};
    }

    cv::Point2d mean_center(0.0, 0.0);
    for (const auto& record : track) {
        mean_center += record_center(record);
        radii.push_back(record_radius(record));
        confidences.push_back(record.confidence);
    }
    mean_center *= 1.0 / static_cast<double>(track.size());

    for (const auto& record : track) {
        centers.push_back(cv::norm(record_center(record) - mean_center));
    }

    details.center_jitter = score_small_is_good(mean_value(centers), 2.0);
    details.radius_jitter = score_small_is_good(stddev_value(radii, mean_value(radii)), 1.0);
    details.confidence_jitter = score_small_is_good(stddev_value(confidences, mean_value(confidences)), 0.15);
    details.detection_continuity = track.size() > 1 ? 1.0 : 0.0;

    StabilityMetricResult result;
    result.score = 0.40 * details.center_jitter
        + 0.25 * details.radius_jitter
        + 0.20 * details.confidence_jitter
        + 0.15 * details.detection_continuity;
    result.passed = result.score >= 0.6;
    return result;
}

StabilityMetricResult evaluate_topology_stability(const std::vector<CircleDetectionRecord>& track,
    const std::vector<NeighborSnapshot>& neighbors, TopologyStabilityDetails& details)
{
    if (track.empty() || neighbors.size() != track.size()) {
        return {};
    }

    std::vector<double> neighbor_counts;
    std::vector<double> mean_distances;
    std::vector<double> mean_angles;

    for (size_t i = 0; i < neighbors.size(); ++i) {
        const cv::Point2d center = record_center(track[i]);
        neighbor_counts.push_back(static_cast<double>(neighbors[i].neighbor_centers.size()));

        std::vector<double> distances;
        std::vector<double> angles;
        for (const auto& neighbor_center : neighbors[i].neighbor_centers) {
            const cv::Point2d diff = neighbor_center - center;
            distances.push_back(cv::norm(diff));
            angles.push_back(std::atan2(diff.y, diff.x));
        }
        mean_distances.push_back(mean_value(distances));
        mean_angles.push_back(circular_mean_angle(angles));
    }

    details.neighbor_count_consistency = score_small_is_good(stddev_value(neighbor_counts, mean_value(neighbor_counts)), 0.5);
    details.relative_distance_consistency = score_small_is_good(stddev_value(mean_distances, mean_value(mean_distances)), 2.0);
    details.relative_angle_consistency = score_small_is_good(circular_stddev(mean_angles), 0.5);
    details.neighbor_identity_stability = track.size() > 1 ? 1.0 : 0.0;

    StabilityMetricResult result;
    result.score = 0.35 * details.neighbor_count_consistency
        + 0.35 * details.relative_distance_consistency
        + 0.30 * details.relative_angle_consistency;
    result.passed = result.score >= 0.6;
    return result;
}

StabilityMetricResult evaluate_brightness_consistency(const std::vector<CircleDetectionRecord>& track,
    BrightnessStabilityDetails& details)
{
    if (track.empty()) {
        return {};
    }

    std::vector<double> contrasts;
    std::vector<double> centers;
    std::vector<double> radii;
    cv::Point2d mean_center(0.0, 0.0);

    for (const auto& record : track) {
        mean_center += record_center(record);
        contrasts.push_back(record.brightness_stats.local_contrast);
        radii.push_back(record_radius(record));
    }
    mean_center *= 1.0 / static_cast<double>(track.size());

    for (const auto& record : track) {
        centers.push_back(cv::norm(record_center(record) - mean_center));
    }

    const double contrast_mean = mean_value(contrasts);
    const double contrast_stddev = stddev_value(contrasts, contrast_mean);

    details.brightness_variation_coverage = clamp01(contrast_stddev / std::max(1.0, contrast_mean));
    details.center_drift_under_brightness_change = score_small_is_good(mean_value(centers), 2.0);
    details.radius_drift_under_brightness_change = score_small_is_good(stddev_value(radii, mean_value(radii)), 1.0);
    details.contrast_retention = clamp01(contrast_mean / std::max(1.0, contrast_mean + contrast_stddev));

    StabilityMetricResult result;
    result.score = 0.10 * details.brightness_variation_coverage
        + 0.40 * details.center_drift_under_brightness_change
        + 0.30 * details.radius_drift_under_brightness_change
        + 0.20 * details.contrast_retention;
    result.passed = result.score >= 0.6;
    return result;
}

CircleStabilityReport evaluate_circle_stability(const std::vector<CircleDetectionRecord>& track,
    const std::vector<NeighborSnapshot>& neighbors, const detector_config& config)
{
    CircleStabilityReport report;
    if (track.empty()) {
        return report;
    }

    report.shape = evaluate_shape_stability(track.back(), config, report.shape_details);
    report.temporal = evaluate_temporal_stability(track, report.temporal_details);
    report.topology = evaluate_topology_stability(track, neighbors, report.topology_details);
    report.brightness = evaluate_brightness_consistency(track, report.brightness_details);
    report.overall_score = 0.30 * report.shape.score
        + 0.25 * report.temporal.score
        + 0.20 * report.topology.score
        + 0.25 * report.brightness.score;
    report.passed = report.overall_score >= 0.6;
    return report;
}

ImageStabilityReport evaluate_image_stability(const ImageDetectionSummary& summary,
    const std::vector<std::vector<CircleDetectionRecord>>& tracks,
    const std::vector<std::vector<NeighborSnapshot>>& neighbors, const detector_config& config)
{
    ImageStabilityReport report;

    if (summary.records.empty()) {
        return report;
    }

    double shape_sum = 0.0;
    for (const auto& record : summary.records) {
        ShapeStabilityDetails details;
        shape_sum += evaluate_shape_stability(record, config, details).score;
    }
    report.shape.score = shape_sum / static_cast<double>(summary.records.size());
    report.shape.passed = report.shape.score >= 0.6;

    if (!tracks.empty()) {
        double temporal_sum = 0.0;
        double topology_sum = 0.0;
        double brightness_sum = 0.0;
        const size_t usable_count = std::min(tracks.size(), neighbors.size());
        for (size_t i = 0; i < usable_count; ++i) {
            TemporalStabilityDetails temporal_details;
            TopologyStabilityDetails topology_details;
            BrightnessStabilityDetails brightness_details;
            temporal_sum += evaluate_temporal_stability(tracks[i], temporal_details).score;
            topology_sum += evaluate_topology_stability(tracks[i], neighbors[i], topology_details).score;
            brightness_sum += evaluate_brightness_consistency(tracks[i], brightness_details).score;
        }
        if (usable_count > 0) {
            report.temporal.score = temporal_sum / static_cast<double>(usable_count);
            report.topology.score = topology_sum / static_cast<double>(usable_count);
            report.brightness.score = brightness_sum / static_cast<double>(usable_count);
        }
    }

    report.temporal.passed = report.temporal.score >= 0.6;
    report.topology.passed = report.topology.score >= 0.6;
    report.brightness.passed = report.brightness.score >= 0.6;
    report.overall_score = 0.30 * report.shape.score
        + 0.25 * report.temporal.score
        + 0.20 * report.topology.score
        + 0.25 * report.brightness.score;
    report.passed = report.overall_score >= 0.6;
    return report;
}
