#pragma once

#include "super_edge_detector.h"

#include <vector>

struct ShapeStabilityDetails
{
    double valid_ray_ratio = 0.0;
    double sharpness_consistency = 0.0;
    double fit_cost_quality = 0.0;
    double angular_coverage = 0.0;
    double contour_roundness = 0.0;
};

struct TemporalStabilityDetails
{
    double center_jitter = 0.0;
    double radius_jitter = 0.0;
    double confidence_jitter = 0.0;
    double detection_continuity = 0.0;
};

struct TopologyStabilityDetails
{
    double neighbor_count_consistency = 0.0;
    double relative_distance_consistency = 0.0;
    double relative_angle_consistency = 0.0;
    double neighbor_identity_stability = 0.0;
};

struct BrightnessStabilityDetails
{
    double brightness_variation_coverage = 0.0;
    double center_drift_under_brightness_change = 0.0;
    double radius_drift_under_brightness_change = 0.0;
    double contrast_retention = 0.0;
};

struct StabilityMetricResult
{
    double score = 0.0;
    bool passed = false;
};

struct CircleStabilityReport
{
    StabilityMetricResult shape;
    StabilityMetricResult temporal;
    StabilityMetricResult topology;
    StabilityMetricResult brightness;
    ShapeStabilityDetails shape_details;
    TemporalStabilityDetails temporal_details;
    TopologyStabilityDetails topology_details;
    BrightnessStabilityDetails brightness_details;
    double overall_score = 0.0;
    bool passed = false;
};

struct ImageStabilityReport
{
    StabilityMetricResult shape;
    StabilityMetricResult temporal;
    StabilityMetricResult topology;
    StabilityMetricResult brightness;
    double overall_score = 0.0;
    bool passed = false;
};

struct NeighborSnapshot
{
    std::vector<cv::Point2d> neighbor_centers;
};

StabilityMetricResult evaluate_shape_stability(const CircleDetectionRecord& record, const detector_config& config,
    ShapeStabilityDetails& details);
StabilityMetricResult evaluate_temporal_stability(const std::vector<CircleDetectionRecord>& track,
    TemporalStabilityDetails& details);
StabilityMetricResult evaluate_topology_stability(const std::vector<CircleDetectionRecord>& track,
    const std::vector<NeighborSnapshot>& neighbors, TopologyStabilityDetails& details);
StabilityMetricResult evaluate_brightness_consistency(const std::vector<CircleDetectionRecord>& track,
    BrightnessStabilityDetails& details);
CircleStabilityReport evaluate_circle_stability(const std::vector<CircleDetectionRecord>& track,
    const std::vector<NeighborSnapshot>& neighbors, const detector_config& config);
ImageStabilityReport evaluate_image_stability(const ImageDetectionSummary& summary,
    const std::vector<std::vector<CircleDetectionRecord>>& tracks,
    const std::vector<std::vector<NeighborSnapshot>>& neighbors, const detector_config& config);
