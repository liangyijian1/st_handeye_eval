#include "super_edge_detector.h"

#include <iostream>
#include <string>


int main(int argc, char** argv)
{
    detector_config config;
    config.num_directions = 20;
    config.radial_margin = 7.0;
    config.radial_step = 0.2;


    config.save_plots = false;   // 每条射线的拟合曲线图
    config.save_crops = false;   // 每个圆的裁剪放大图


    super_edge_detector edge_detector("C:\\Users\\yjliang\\Downloads\\st_handeye_eval\\st_handeye_eval\\images", config);
    edge_detector.detect_edges();

    std::cout << "\nProcessing complete!" << std::endl;
    return 0;
}

