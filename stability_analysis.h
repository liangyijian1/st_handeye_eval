#pragma once

#include <opencv2/core.hpp>

#include <optional>
#include <string>
#include <vector>

// 单个圆形标志点在一帧图像/点云中的观测结果。
// 该结构只描述“已经检测出的标志点”，不负责检测、匹配或稳定性评分。
struct MarkerObservation
{
    // 标志点编号。未知或尚未完成跨帧匹配时保持 -1。
    int id = -1;

    // 标志点在原始图像中的二维中心坐标，单位为像素。
    cv::Point2d center2d;

    // 标志点映射到点云后的三维坐标，单位由上游点云坐标系决定。
    // 若当前阶段只有图像观测、还没有完成 2D 到 3D 映射，则为空。
    std::optional<cv::Point3d> center3d;

    // 图像中的标志点半径，单位为像素。
    double radius = 0.0;

    // 上游检测阶段给出的单点置信度，建议范围为 [0, 1]。
    double confidence = 0.0;
};

// 一帧数据中所有标志点观测的集合。
// 点云拼接流程可把每次扫描/每张图像整理成一个 FrameMarkerSet 后送入稳定性分析。
struct FrameMarkerSet
{
    // 帧编号，用于日志、调试和与外部点云帧关联。
    std::string frame_id;

    // 时间戳。单位由调用方约定，可为秒、毫秒或帧序号。
    double timestamp = 0.0;

    // 当前帧内检测到的所有圆形标志点。
    std::vector<MarkerObservation> markers;
};

// 稳定性分析配置。
// 当前只保留拓扑和时域两类判断最基础的阈值入口，具体评分算法后续补充。
struct StabilityAnalyzerConfig
{
    // 拓扑判断所需的最少标志点数量。少于该数量时，后续算法通常无法形成可靠邻接/距离约束。
    int min_marker_count_for_topology = 3;

    // 时域判断使用的历史窗口长度，表示最多参考多少帧 marker 观测。
    int temporal_window_size = 5;
};

// 单帧拓扑稳定性结果。
// 后续可在这里扩展邻接一致性、相对距离一致性、缺点/误检判断等指标。
struct TopologyStabilityResult
{
    // 当前拓扑算法是否已经实现。骨架阶段固定为 false。
    bool implemented = false;

    // 当前帧拓扑关系是否稳定。算法未实现时固定为 false。
    bool stable = false;

    // 拓扑稳定性评分，建议范围为 [0, 1]。算法未实现时为 0。
    double score = 0.0;

    // 参与拓扑判断的当前帧标志点数量。
    int marker_count = 0;

    // 诊断信息，用于说明未实现、数据不足或评分失败原因。
    std::string message;
};

// 多帧时域稳定性结果。
// 后续可在这里扩展轨迹连续性、速度/位移突变、出现消失频率等指标。
struct TemporalStabilityResult
{
    // 当前时域算法是否已经实现。骨架阶段固定为 false。
    bool implemented = false;

    // 多帧观测在时间方向是否稳定。算法未实现时固定为 false。
    bool stable = false;

    // 时域稳定性评分，建议范围为 [0, 1]。算法未实现时为 0。
    double score = 0.0;

    // 参与时域判断的帧数。
    int frame_count = 0;

    // 具有有效 id 的不同标志点轨迹数量。
    int marker_track_count = 0;

    // 诊断信息，用于说明未实现、数据不足或评分失败原因。
    std::string message;
};

// 稳定性综合结果。
// 点云拼接前可根据该结构决定 marker 是否可参与刚体配准或作为 ICP 初值。
struct StabilityAnalysisResult
{
    // 当前帧或最近一帧的拓扑稳定性结果。
    TopologyStabilityResult topology;

    // 输入帧序列的时域稳定性结果。
    TemporalStabilityResult temporal;

    // 综合评分，建议范围为 [0, 1]。骨架阶段为 0。
    double combined_score = 0.0;

    // 是否建议把这些标志点用于点云配准。骨架阶段固定为 false。
    bool usable_for_registration = false;
};

// 圆形标志点稳定性分析器。
// 当前类只提供拓扑和时域稳定性接口骨架，不依赖 super_edge_detector，也不修改检测流程。
class StabilityAnalyzer
{
public:
    StabilityAnalyzer() = default;
    explicit StabilityAnalyzer(StabilityAnalyzerConfig config);

    // 读取当前分析配置，便于外部确认阈值和窗口参数。
    const StabilityAnalyzerConfig& config() const;

    // 分析单帧内标志点之间的拓扑稳定性。
    TopologyStabilityResult analyzeTopology(const FrameMarkerSet& frame) const;

    // 分析多帧标志点观测在时间方向的稳定性。
    TemporalStabilityResult analyzeTemporal(const std::vector<FrameMarkerSet>& frames) const;

    // 同时运行拓扑和时域稳定性分析，并返回综合结果。
    StabilityAnalysisResult analyze(const std::vector<FrameMarkerSet>& frames) const;

private:
    StabilityAnalyzerConfig config_;
};
