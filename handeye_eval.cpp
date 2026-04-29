#include "handeye_eval.h"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>

namespace {

std::vector<double> extractNumbers(const std::string& text)
{
    static const std::regex numberPattern(R"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)");
    std::vector<double> values;
    for (auto it = std::sregex_iterator(text.begin(), text.end(), numberPattern);
        it != std::sregex_iterator(); ++it) {
        values.push_back(std::stod(it->str()));
    }
    return values;
}

std::string readTextFile(const std::filesystem::path& path)
{
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("failed to open " + path.string());
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    return buffer.str();
}

double extractScalarField(const std::string& text, const std::string& key)
{
    const std::regex fieldPattern(key + R"(\s*:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?))");
    std::smatch match;
    if (!std::regex_search(text, match, fieldPattern)) {
        throw std::runtime_error("missing scalar field in camera_info.yaml: " + key);
    }
    return std::stod(match[1].str());
}

std::vector<double> extractSectionData(const std::string& text, const std::string& section)
{
    const std::regex sectionPattern(section + R"(:[\s\S]*?data:\s*\[([^\]]+)\])");
    std::smatch match;
    if (!std::regex_search(text, match, sectionPattern)) {
        throw std::runtime_error("missing data field in camera_info.yaml section: " + section);
    }
    return extractNumbers(match[1].str());
}

std::string threeDigitId(int index)
{
    std::ostringstream stream;
    stream << std::setw(3) << std::setfill('0') << index;
    return stream.str();
}

bool isFiniteMat(const cv::Mat& matrix)
{
    if (matrix.empty()) {
        return false;
    }
    for (int row = 0; row < matrix.rows; ++row) {
        for (int col = 0; col < matrix.cols; ++col) {
            if (!std::isfinite(matrix.at<double>(row, col))) {
                return false;
            }
        }
    }
    return true;
}

cv::Mat identity4()
{
    return cv::Mat::eye(4, 4, CV_64F);
}

cv::Mat invertRigidOrGeneral(const cv::Mat& matrix)
{
    cv::Mat inverse;
    if (!cv::invert(matrix, inverse, cv::DECOMP_SVD)) {
        throw std::runtime_error("pose matrix is not invertible");
    }
    return inverse;
}

double rotationAngleDeg(const cv::Mat& rotation)
{
    const double trace = rotation.at<double>(0, 0) + rotation.at<double>(1, 1) + rotation.at<double>(2, 2);
    const double cosAngle = std::clamp((trace - 1.0) * 0.5, -1.0, 1.0);
    return std::acos(cosAngle) * 180.0 / CV_PI;
}

double determinant3x3(const cv::Mat& matrix)
{
    return cv::determinant(matrix(cv::Rect(0, 0, 3, 3)));
}

} // namespace

HandEyeEvaluator::HandEyeEvaluator(EvalConfig config)
    : config_(std::move(config))
{
}

EvaluationReport HandEyeEvaluator::evaluate(const std::filesystem::path& root) const
{
    EvaluationReport report;
    report.cameraInfo = loadCameraInfo(root / "camera_info.yaml");

    const std::filesystem::path dataDir = root / config_.dataSubdir;
    for (int index = 0; index <= 17; ++index) {
        const std::string id = threeDigitId(index);
        report.frames.push_back(evaluateFrame(dataDir / (id + "_image.jpg"),
            dataDir / (id + "_pose.csv"), report.cameraInfo));
    }

    report.modes.push_back(calibrateMode(report.frames, "pose.csv as gripper-to-base", false));
    report.modes.push_back(calibrateMode(report.frames, "inverse(pose.csv) as gripper-to-base", true));
    return report;
}

bool HandEyeEvaluator::selfTest(const std::filesystem::path& root, std::ostream& output) const
{
    bool ok = true;
    const auto expect = [&](bool condition, const std::string& message) {
        output << (condition ? "[PASS] " : "[FAIL] ") << message << '\n';
        ok = ok && condition;
    };

    try {
        const CameraInfo cameraInfo = loadCameraInfo(root / "camera_info.yaml");
        expect(cameraInfo.imageWidth == 1928, "camera width is 1928");
        expect(cameraInfo.imageHeight == 1448, "camera height is 1448");
        expect(cameraInfo.cameraMatrix.rows == 3 && cameraInfo.cameraMatrix.cols == 3, "camera matrix is 3x3");
        expect(cameraInfo.distCoeffs.total() == 5, "distortion vector has 5 coefficients");
        expect(std::abs(cameraInfo.cameraMatrix.at<double>(0, 0) - 1677.769190486581) < 1e-9, "camera fx parsed correctly");
        expect(std::abs(cameraInfo.cameraMatrix.at<double>(1, 1) - 1686.122070709732) < 1e-9, "camera fy parsed correctly");
        expect(std::abs(cameraInfo.distCoeffs.at<double>(0, 0) + 0.3299099196112457) < 1e-12,
            "first distortion coefficient parsed correctly");

        const auto objectPoints = makeObjectPoints();
        expect(objectPoints.size() == 44, "asymmetric board produces 44 object points");
        expect(std::abs(objectPoints[0].x - 0.0f) < 1e-6f, "first object point x");
        expect(std::abs(objectPoints[4].x - static_cast<float>(config_.boardLengthMeters)) < 1e-6f,
            "second row starts with half-offset square size");

        const cv::Mat pose = loadPoseMatrix(root / config_.dataSubdir / "000_pose.csv");
        const cv::Mat roundTrip = pose * invertRigidOrGeneral(pose);
        expect(cv::norm(roundTrip - identity4(), cv::NORM_INF) < 1e-8, "pose inverse round trip");

        const cv::Mat image = cv::imread((root / config_.dataSubdir / "000_image.jpg").string(), cv::IMREAD_UNCHANGED);
        const auto detection = PackAlgoCircleGridDetector(config_.detector).detect(image);
        expect(detection.found, "frame 000 circle grid is detected");
        expect(detection.centers.size() == 44, "frame 000 has 44 detected centers");
    }
    catch (const std::exception& ex) {
        output << "[FAIL] exception: " << ex.what() << '\n';
        ok = false;
    }

    return ok;
}

void HandEyeEvaluator::writeReport(const EvaluationReport& report, std::ostream& output) const
{
    output << "st_handeye_eval standalone report\n";
    output << "camera: " << report.cameraInfo.imageWidth << "x" << report.cameraInfo.imageHeight << "\n";
    output << "board: cols=" << config_.detector.boardCols
        << " rows=" << config_.detector.boardRows
        << " asymmetric=" << (config_.detector.asymmetricGrid ? "true" : "false")
        << " boardLengthMeters=" << config_.boardLengthMeters << "\n\n";

    output << "camera_matrix:\n" << report.cameraInfo.cameraMatrix << "\n";
    output << "dist_coeffs:\n" << report.cameraInfo.distCoeffs << "\n\n";

    int detectedCount = 0;
    int pnpCount = 0;
    output << "frames:\n";
    for (const auto& frame : report.frames) {
        detectedCount += frame.circleFound ? 1 : 0;
        pnpCount += frame.pnpSucceeded ? 1 : 0;
        output << "  " << frame.frameId
            << " image=" << (frame.imageLoaded ? "ok" : "fail")
            << " pose=" << (frame.poseLoaded ? "ok" : "fail")
            << " keypoints=" << frame.keypointCount
            << " centers=" << frame.centerCount
            << " circle=" << (frame.circleFound ? "ok" : "fail")
            << " pnp=" << (frame.pnpSucceeded ? "ok" : "fail")
            << " reprojRms=" << std::fixed << std::setprecision(6) << frame.reprojectionRms
            << " message=" << frame.message << "\n";
    }
    output << "\nsummary: detectedFrames=" << detectedCount
        << " pnpFrames=" << pnpCount
        << " totalFrames=" << report.frames.size() << "\n\n";

    output << "hand_eye_modes:\n";
    for (const auto& mode : report.modes) {
        output << "mode: " << mode.modeName << "\n";
        output << "  success=" << (mode.succeeded ? "true" : "false")
            << " validFrameCount=" << mode.validFrameCount
            << " message=" << mode.message << "\n";
        if (mode.succeeded) {
            output << "  cameraToGripper:\n" << mode.cameraToGripper << "\n";
            output << "  det(R)=" << determinant3x3(mode.cameraToGripper) << "\n";
            output << "  positionStddev=" << mode.positionStddev
                << " positionMaxError=" << mode.positionMaxError << "\n";
            output << "  rotationStddevDeg=" << mode.rotationStddevDeg
                << " rotationMaxErrorDeg=" << mode.rotationMaxErrorDeg << "\n";
        }
        output << "\n";
    }
}

CameraInfo HandEyeEvaluator::loadCameraInfo(const std::filesystem::path& path) const
{
    const std::string text = readTextFile(path);

    CameraInfo info;
    info.imageWidth = static_cast<int>(extractScalarField(text, "image_width"));
    info.imageHeight = static_cast<int>(extractScalarField(text, "image_height"));

    const std::vector<double> cameraMatrixValues = extractSectionData(text, "camera_matrix");
    const std::vector<double> distortionValues = extractSectionData(text, "distortion_coefficients");
    if (cameraMatrixValues.size() != 9) {
        throw std::runtime_error("camera_matrix data must contain 9 values");
    }
    if (distortionValues.size() != 5) {
        throw std::runtime_error("distortion_coefficients data must contain 5 values");
    }

    info.cameraMatrix = cv::Mat::eye(3, 3, CV_64F);
    for (int i = 0; i < 9; ++i) {
        info.cameraMatrix.at<double>(i / 3, i % 3) = cameraMatrixValues[i];
    }

    info.distCoeffs = cv::Mat(1, 5, CV_64F);
    for (int i = 0; i < 5; ++i) {
        info.distCoeffs.at<double>(0, i) = distortionValues[i];
    }
    return info;
}

cv::Mat HandEyeEvaluator::loadPoseMatrix(const std::filesystem::path& path) const
{
    const std::vector<double> values = extractNumbers(readTextFile(path));
    if (values.size() != 16) {
        throw std::runtime_error("pose file must contain 16 numeric values: " + path.string());
    }

    cv::Mat pose(4, 4, CV_64F);
    for (int i = 0; i < 16; ++i) {
        pose.at<double>(i / 4, i % 4) = values[i];
    }
    if (!isFiniteMat(pose)) {
        throw std::runtime_error("pose file contains non-finite values: " + path.string());
    }
    return pose;
}

std::vector<cv::Point3f> HandEyeEvaluator::makeObjectPoints() const
{
    std::vector<cv::Point3f> objectPoints;
    objectPoints.reserve(static_cast<size_t>(config_.detector.boardCols * config_.detector.boardRows));
    for (int row = 0; row < config_.detector.boardRows; ++row) {
        for (int col = 0; col < config_.detector.boardCols; ++col) {
            if (config_.detector.asymmetricGrid) {
                objectPoints.emplace_back(
                    static_cast<float>((2 * col + row % 2) * config_.boardLengthMeters),
                    static_cast<float>(row * config_.boardLengthMeters),
                    0.0f);
            }
            else {
                objectPoints.emplace_back(
                    static_cast<float>(col * config_.boardLengthMeters),
                    static_cast<float>(row * config_.boardLengthMeters),
                    0.0f);
            }
        }
    }
    return objectPoints;
}

FrameEvaluation HandEyeEvaluator::evaluateFrame(const std::filesystem::path& imagePath,
    const std::filesystem::path& posePath, const CameraInfo& cameraInfo) const
{
    FrameEvaluation frame;
    frame.frameId = imagePath.stem().string().substr(0, 3);
    try {
        frame.poseMatrix = loadPoseMatrix(posePath);
        frame.poseLoaded = true;
    }
    catch (const std::exception& ex) {
        frame.message = ex.what();
        return frame;
    }

    const cv::Mat image = cv::imread(imagePath.string(), cv::IMREAD_UNCHANGED);
    frame.imageLoaded = !image.empty();
    if (!frame.imageLoaded) {
        frame.message = "image load failed";
        return frame;
    }

    const auto detection = PackAlgoCircleGridDetector(config_.detector).detect(image);
    frame.keypointCount = detection.keypointCount;
    frame.centerCount = static_cast<int>(detection.centers.size());
    frame.circleFound = detection.found;
    if (!detection.found) {
        frame.message = detection.message;
        return frame;
    }

    const auto objectPoints = makeObjectPoints();
    cv::Mat rvec;
    cv::Mat tvec;
    try {
        frame.pnpSucceeded = cv::solvePnP(objectPoints, detection.centers, cameraInfo.cameraMatrix,
            cameraInfo.distCoeffs, rvec, tvec, false, cv::SOLVEPNP_ITERATIVE);
    }
    catch (const cv::Exception& ex) {
        frame.message = ex.what();
        return frame;
    }

    if (!frame.pnpSucceeded) {
        frame.message = "solvePnP failed";
        return frame;
    }

    cv::Mat rotation;
    cv::Rodrigues(rvec, rotation);
    frame.targetToCamera = makeTransform(rotation, tvec);
    frame.reprojectionRms = computeReprojectionRms(objectPoints, detection.centers, rvec, tvec, cameraInfo);
    frame.message = "ok";
    return frame;
}

HandEyeModeEvaluation HandEyeEvaluator::calibrateMode(const std::vector<FrameEvaluation>& frames,
    const std::string& modeName, bool invertRobotPose) const
{
    HandEyeModeEvaluation mode;
    mode.modeName = modeName;

    std::vector<cv::Mat> gripperToBaseRotations;
    std::vector<cv::Mat> gripperToBaseTranslations;
    std::vector<cv::Mat> targetToCameraRotations;
    std::vector<cv::Mat> targetToCameraTranslations;
    std::vector<cv::Mat> gripperToBaseMatrices;
    std::vector<cv::Mat> targetToCameraMatrices;

    for (const auto& frame : frames) {
        if (!frame.pnpSucceeded || frame.targetToCamera.empty() || frame.poseMatrix.empty()) {
            continue;
        }

        cv::Mat gripperToBase = invertRobotPose ? invertRigidOrGeneral(frame.poseMatrix) : frame.poseMatrix.clone();
        gripperToBase.convertTo(gripperToBase, CV_64F);

        gripperToBaseRotations.push_back(gripperToBase(cv::Rect(0, 0, 3, 3)).clone());
        gripperToBaseTranslations.push_back(gripperToBase(cv::Rect(3, 0, 1, 3)).clone());
        targetToCameraRotations.push_back(frame.targetToCamera(cv::Rect(0, 0, 3, 3)).clone());
        targetToCameraTranslations.push_back(frame.targetToCamera(cv::Rect(3, 0, 1, 3)).clone());
        gripperToBaseMatrices.push_back(gripperToBase);
        targetToCameraMatrices.push_back(frame.targetToCamera);
    }

    mode.validFrameCount = static_cast<int>(targetToCameraMatrices.size());
    if (mode.validFrameCount < 3) {
        mode.message = "fewer than 3 valid frames";
        return mode;
    }

    cv::Mat rCameraToGripper;
    cv::Mat tCameraToGripper;
    try {
        cv::calibrateHandEye(gripperToBaseRotations, gripperToBaseTranslations,
            targetToCameraRotations, targetToCameraTranslations,
            rCameraToGripper, tCameraToGripper, cv::CALIB_HAND_EYE_TSAI);
    }
    catch (const cv::Exception& ex) {
        mode.message = ex.what();
        return mode;
    }

    mode.cameraToGripper = makeTransform(rCameraToGripper, tCameraToGripper);
    if (!isFiniteMat(mode.cameraToGripper)) {
        mode.message = "hand-eye result contains non-finite values";
        return mode;
    }

    computeConsistency(targetToCameraMatrices, gripperToBaseMatrices, mode.cameraToGripper, mode);
    mode.succeeded = true;
    mode.message = "ok";
    return mode;
}

cv::Mat HandEyeEvaluator::makeTransform(const cv::Mat& rotation, const cv::Mat& translation) const
{
    cv::Mat transform = identity4();
    cv::Mat rotation64;
    cv::Mat translation64;
    rotation.convertTo(rotation64, CV_64F);
    translation.convertTo(translation64, CV_64F);

    rotation64.copyTo(transform(cv::Rect(0, 0, 3, 3)));
    cv::Mat translationColumn = translation64.reshape(1, 3);
    translationColumn.copyTo(transform(cv::Rect(3, 0, 1, 3)));
    return transform;
}

double HandEyeEvaluator::computeReprojectionRms(const std::vector<cv::Point3f>& objectPoints,
    const std::vector<cv::Point2f>& imagePoints, const cv::Mat& rvec, const cv::Mat& tvec,
    const CameraInfo& cameraInfo) const
{
    std::vector<cv::Point2f> projected;
    cv::projectPoints(objectPoints, rvec, tvec, cameraInfo.cameraMatrix, cameraInfo.distCoeffs, projected);
    if (projected.size() != imagePoints.size() || projected.empty()) {
        return 0.0;
    }

    double sumSquared = 0.0;
    for (size_t i = 0; i < projected.size(); ++i) {
        const cv::Point2f diff = projected[i] - imagePoints[i];
        sumSquared += static_cast<double>(diff.x * diff.x + diff.y * diff.y);
    }
    return std::sqrt(sumSquared / static_cast<double>(projected.size()));
}

void HandEyeEvaluator::computeConsistency(const std::vector<cv::Mat>& targetToCameraMatrices,
    const std::vector<cv::Mat>& gripperToBaseMatrices, const cv::Mat& cameraToGripper,
    HandEyeModeEvaluation& mode) const
{
    std::vector<cv::Point3d> positions;
    std::vector<double> rotationAngles;
    positions.reserve(targetToCameraMatrices.size());
    rotationAngles.reserve(targetToCameraMatrices.size());

    for (size_t i = 0; i < targetToCameraMatrices.size(); ++i) {
        const cv::Mat targetToBase = gripperToBaseMatrices[i] * cameraToGripper * targetToCameraMatrices[i];
        positions.emplace_back(
            targetToBase.at<double>(0, 3),
            targetToBase.at<double>(1, 3),
            targetToBase.at<double>(2, 3));
        rotationAngles.push_back(rotationAngleDeg(targetToBase(cv::Rect(0, 0, 3, 3))));
    }

    cv::Point3d meanPosition(0.0, 0.0, 0.0);
    for (const auto& position : positions) {
        meanPosition += position;
    }
    meanPosition *= 1.0 / static_cast<double>(positions.size());

    double positionVariance = 0.0;
    double positionMaxError = 0.0;
    for (const auto& position : positions) {
        const cv::Point3d diff = position - meanPosition;
        const double norm = std::sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
        positionVariance += norm * norm;
        positionMaxError = std::max(positionMaxError, norm);
    }
    mode.positionStddev = std::sqrt(positionVariance / static_cast<double>(positions.size()));
    mode.positionMaxError = positionMaxError;

    const double meanRotation = std::accumulate(rotationAngles.begin(), rotationAngles.end(), 0.0)
        / static_cast<double>(rotationAngles.size());
    double rotationVariance = 0.0;
    double rotationMaxError = 0.0;
    for (double angle : rotationAngles) {
        const double diff = angle - meanRotation;
        rotationVariance += diff * diff;
        rotationMaxError = std::max(rotationMaxError, std::abs(diff));
    }
    mode.rotationStddevDeg = std::sqrt(rotationVariance / static_cast<double>(rotationAngles.size()));
    mode.rotationMaxErrorDeg = rotationMaxError;
}
