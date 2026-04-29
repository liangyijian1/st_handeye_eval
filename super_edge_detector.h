#pragma once
#include <opencv2/opencv.hpp>


class super_edge_detector
{
private:
    /* data */
public:
    super_edge_detector(/* args */);
    ~super_edge_detector();

private:
    bool detect_edges(const cv::Mat& image, std::vector<cv::Point>& edge_points);
};

super_edge_detector::super_edge_detector(/* args */)
{
}

super_edge_detector::~super_edge_detector()
{
}
