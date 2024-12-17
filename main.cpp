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
    cv::resize(outImg,outImg,cv::Size(640,640),cv::INTER_LINEAR);
    objTool.HWC2NormalCHW(outImg, data);
    std::cout << data[0] << std::endl;
    cv::Mat res;
    inferenceTool infTool("./data/yolov9.trt");
    infTool.run(data,res);

    cv::imwrite("./data/test.png",outImg);
    // std::cout << r << std::endl;
}
