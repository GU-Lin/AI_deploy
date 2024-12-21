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

    objTool.HWC2NormalCHW(outImg, data);
    std::cout << data[0] << std::endl;
    cv::Mat res;
    TRTInferenceTool infTool("./data/yolov9-c-converted.trt");
    infTool.run(data,res);
    objTool.postprocess(res);
    objTool.draw(img,img,objTool.m_PredBox_vector);
    cv::imwrite("./data/test.png",img);
    // std::cout << r << std::endl;
}
