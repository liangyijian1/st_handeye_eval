#include "stability_analysis.h"

#include <unordered_map>

StabilityAnalyzer::StabilityAnalyzer(StabilityAnalyzerConfig config)
    : config_(config)
{
}

const StabilityAnalyzerConfig& StabilityAnalyzer::config() const
{
    return config_;
}

TopologyStabilityResult StabilityAnalyzer::analyzeTopology(const FrameMarkerSet& frame) const
{
    TopologyStabilityResult result;
    result.marker_count = static_cast<int>(frame.markers.size());
    result.message = "topology stability analysis is not implemented";
    return result;
}

TemporalStabilityResult StabilityAnalyzer::analyzeTemporal(const std::vector<FrameMarkerSet>& frames) const
{
    TemporalStabilityResult result;
    result.frame_count = static_cast<int>(frames.size());

    // 骨架阶段只统计带有效 id 的轨迹数量，方便后续接入时验证输入数据是否正确传入。
    std::unordered_map<int, int> track_counts;
    for (const auto& frame : frames) {
        for (const auto& marker : frame.markers) {
            if (marker.id >= 0) {
                ++track_counts[marker.id];
            }
        }
    }

    result.marker_track_count = static_cast<int>(track_counts.size());
    result.message = "temporal stability analysis is not implemented";
    return result;
}

StabilityAnalysisResult StabilityAnalyzer::analyze(const std::vector<FrameMarkerSet>& frames) const
{
    StabilityAnalysisResult result;
    if (!frames.empty()) {
        // 拓扑稳定性是单帧空间关系判断，综合接口默认使用最近一帧。
        result.topology = analyzeTopology(frames.back());
    } else {
        result.topology.message = "topology stability analysis is not implemented";
    }
    result.temporal = analyzeTemporal(frames);
    return result;
}
