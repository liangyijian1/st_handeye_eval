#pragma once

#include "packalgo_circle_grid_detector.h"

#include <opencv2/core.hpp>

#include <filesystem>
#include <iosfwd>
#include <string>
#include <vector>

struct CameraInfo
{
    int imageWidth = 0;
    int imageHeight = 0;
    cv::Mat cameraMatrix;
    cv::Mat distCoeffs;
};

struct FrameEvaluation
{
    std::string frameId;
    bool imageLoaded = false;
    bool poseLoaded = false;
    bool circleFound = false;
    bool pnpSucceeded = false;
    int keypointCount = 0;
    int centerCount = 0;
    double reprojectionRms = 0.0;
    std::string message;
    cv::Mat targetToCamera;
    cv::Mat poseMatrix;
};

struct HandEyeModeEvaluation
{
    std::string modeName;
    bool succeeded = false;
    int validFrameCount = 0;
    cv::Mat cameraToGripper;
    double positionStddev = 0.0;
    double positionMaxError = 0.0;
    double rotationStddevDeg = 0.0;
    double rotationMaxErrorDeg = 0.0;
    std::string message;
};

struct EvaluationReport
{
    CameraInfo cameraInfo;
    std::vector<FrameEvaluation> frames;
    std::vector<HandEyeModeEvaluation> modes;
};

struct EvalConfig
{
    CircleGridDetectorConfig detector;
    double boardLengthMeters = 0.02;
    std::string dataSubdir = "calib_test";
};

class HandEyeEvaluator
{
public:
    explicit HandEyeEvaluator(EvalConfig config = {});

    EvaluationReport evaluate(const std::filesystem::path& root) const;
    bool selfTest(const std::filesystem::path& root, std::ostream& output) const;
    void writeReport(const EvaluationReport& report, std::ostream& output) const;

private:
    CameraInfo loadCameraInfo(const std::filesystem::path& path) const;
    cv::Mat loadPoseMatrix(const std::filesystem::path& path) const;
    std::vector<cv::Point3f> makeObjectPoints() const;
    FrameEvaluation evaluateFrame(const std::filesystem::path& imagePath,
        const std::filesystem::path& posePath, const CameraInfo& cameraInfo) const;
    HandEyeModeEvaluation calibrateMode(const std::vector<FrameEvaluation>& frames,
        const std::string& modeName, bool invertRobotPose) const;
    cv::Mat makeTransform(const cv::Mat& rotation, const cv::Mat& translation) const;
    double computeReprojectionRms(const std::vector<cv::Point3f>& objectPoints,
        const std::vector<cv::Point2f>& imagePoints, const cv::Mat& rvec, const cv::Mat& tvec,
        const CameraInfo& cameraInfo) const;
    void computeConsistency(const std::vector<cv::Mat>& targetToCameraMatrices,
        const std::vector<cv::Mat>& gripperToBaseMatrices, const cv::Mat& cameraToGripper,
        HandEyeModeEvaluation& mode) const;

    EvalConfig config_;
};

