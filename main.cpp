#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "include/inferenceTool.hpp"
#include "include/utils.hpp"

int main()
{
    // Read image
    cv::Mat img = cv::imread("./data/horses.jpg");
    objectDetection objTool;
    cv::Mat outImg;
    std::vector<float> data;
    objTool.preprocess(img, outImg);
    cv::imwrite("./data/test.png",outImg);
    objTool.HWC2NormalCHW(outImg, data);
    std::cout << data[0] << std::endl;
    cv::Mat res;
    inferenceTool infTool("./data/yolov9-m-converted.trt");
    infTool.run(data,res);
    objTool.postprocess(res);

    // std::cout << r << std::endl;
}
