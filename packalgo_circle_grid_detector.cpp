#include "packalgo_circle_grid_detector.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>

PackAlgoCircleGridDetector::PackAlgoCircleGridDetector(CircleGridDetectorConfig config)
    : config_(std::move(config))
{
}

CircleGridDetectionResult PackAlgoCircleGridDetector::detect(const cv::Mat& image) const
{
    CircleGridDetectionResult result;
    if (config_.boardCols <= 0 || config_.boardRows <= 0) {
        result.message = "invalid board size";
        return result;
    }

    cv::Mat detectorGray;
    if (!makeDetectorGray8U(image, detectorGray)) {
        result.message = "failed to build detector gray image";
        return result;
    }

    cv::SimpleBlobDetector::Params params;
    params.filterByColor = true;
    params.blobColor = static_cast<unsigned char>(config_.blobColor);
    params.minArea = static_cast<float>(config_.minArea);
    params.maxArea = static_cast<float>(config_.maxArea);

    cv::Ptr<cv::FeatureDetector> blobDetector = cv::SimpleBlobDetector::create(params);
    std::vector<cv::KeyPoint> keypoints;
    blobDetector->detect(detectorGray, keypoints);
    result.keypointCount = static_cast<int>(keypoints.size());

    int flags = config_.asymmetricGrid ? cv::CALIB_CB_ASYMMETRIC_GRID : cv::CALIB_CB_SYMMETRIC_GRID;
    if (config_.useClustering) {
        flags |= cv::CALIB_CB_CLUSTERING;
    }

    const cv::Size boardSize(config_.boardCols, config_.boardRows);
    result.found = cv::findCirclesGrid(detectorGray, boardSize, result.centers, flags, blobDetector);
    if (!result.found) {
        result.message = "findCirclesGrid failed";
        result.centers.clear();
        return result;
    }

    result.message = "ok";
    return result;
}

bool PackAlgoCircleGridDetector::convertToGray(const cv::Mat& image, cv::Mat& gray) const
{
    if (image.empty()) {
        return false;
    }

    const int depth = image.depth();
    if (depth != CV_8U && depth != CV_16U && depth != CV_32F && depth != CV_64F) {
        return false;
    }

    const int channels = image.channels();
    if (channels == 1) {
        gray = image;
        return true;
    }

    cv::Mat colorSource32F;
    const cv::Mat* colorSource = &image;
    if (depth == CV_64F) {
        image.convertTo(colorSource32F, CV_32F);
        colorSource = &colorSource32F;
    }

    if (channels == 3) {
        cv::cvtColor(*colorSource, gray, cv::COLOR_BGR2GRAY);
        return true;
    }
    if (channels == 4) {
        cv::cvtColor(*colorSource, gray, cv::COLOR_BGRA2GRAY);
        return true;
    }

    return false;
}

bool PackAlgoCircleGridDetector::makeDetectorGray8U(const cv::Mat& image, cv::Mat& gray8U) const
{
    cv::Mat gray;
    if (!convertToGray(image, gray)) {
        return false;
    }

    switch (gray.depth()) {
    case CV_8U:
        gray8U = gray;
        return true;
    case CV_16U:
    case CV_32F:
    case CV_64F:
    {
        double minValue = 0.0;
        double maxValue = 0.0;
        cv::minMaxLoc(gray, &minValue, &maxValue);
        if (!std::isfinite(minValue) || !std::isfinite(maxValue) || maxValue <= minValue) {
            return false;
        }
        gray.convertTo(gray8U, CV_8U, 255.0 / (maxValue - minValue),
            -255.0 * minValue / (maxValue - minValue));
        return true;
    }
    default:
        return false;
    }
}

