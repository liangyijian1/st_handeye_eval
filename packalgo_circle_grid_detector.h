#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

struct CircleGridDetectorConfig
{
    int boardCols = 4;
    int boardRows = 11;
    bool asymmetricGrid = true;
    bool useClustering = true;
    int blobColor = 0;
    double minArea = 100.0;
    double maxArea = 40000.0;
};

struct CircleGridDetectionResult
{
    bool found = false;
    int keypointCount = 0;
    std::vector<cv::Point2f> centers;
    std::string message;
};

class PackAlgoCircleGridDetector
{
public:
    explicit PackAlgoCircleGridDetector(CircleGridDetectorConfig config = {});

    CircleGridDetectionResult detect(const cv::Mat& image) const;

private:
    bool convertToGray(const cv::Mat& image, cv::Mat& gray) const;
    bool makeDetectorGray8U(const cv::Mat& image, cv::Mat& gray8U) const;

    CircleGridDetectorConfig config_;
};

