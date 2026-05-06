#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "super_edge_detector.h"


int main(int argc, char** argv)
{
    detector_config config;
    super_edge_detector edge_detector("C:\\Users\\yjliang\\Downloads\\st_handeye_eval\\st_handeye_eval\\images", config);
    edge_detector.detect_edges();
    return 0;
}

