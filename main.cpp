#include <iostream>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include "include/trtInferenceTool.hpp"
#include "include/trtCalibrate.hpp"
#include "include/utils.hpp"

std::vector<std::string> loadCalibrationFiles(const std::string& filePath) {
    std::vector<std::string> calibrationFiles;
    std::ifstream file(filePath);

    if (!file.is_open()) {
        std::cerr << "Failed to open the file: " << filePath << std::endl;
        return calibrationFiles;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (!line.empty()) {  // 避免空行
            calibrationFiles.push_back(line);
        }
    }

    file.close();
    return calibrationFiles;
}

void inference()
{
    cv::Mat img = cv::imread("./data/horses.jpg");
    objectDetection objTool;
    cv::Mat outImg;
    std::vector<float> data;
    objTool.preprocess(img, outImg);

    objTool.HWC2NormalCHW(outImg, data);
    std::cout << data[0] << std::endl;
    cv::Mat res;
    TRTInferenceTool infTool("./data/yolov9-c-converted-int8.trt");
    infTool.run(data,res);
    objTool.postprocess(res);
    objTool.draw(img,img,objTool.m_PredBox_vector);
    cv::imwrite("./data/test.png",img);
}

void calibrate()
{
    std::string filePath = "calibration.txt";
    std::vector<std::string> calFile = loadCalibrationFiles(filePath);

    MyInt8Calibrator *Calibrator = new MyInt8Calibrator(calFile, 640, 640);

    TRTCalibrateTool calTool;
    calTool.setConfig(*Calibrator);
    calTool.buildSerial("./data/yolov9-c-converted.onnx","./data/yolov9-c-converted-int8.engine");
}



int main()
{
    calibrate();
    inference();
    return 0;
}
