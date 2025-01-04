#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>
#include <common/buffers.h>
#include <fstream>
#include <memory>
#include <fstream>

using namespace nvinfer1;
class Logger : public ILogger {
public:
    void log(ILogger::Severity severity, const char* msg) noexcept override {
        // Only print messages for errors
        // if (severity <= ILogger::Severity::kERROR) {
        //     std::cerr << msg << std::endl;
        // }
        if (severity <= Severity::kINFO) {
            std::cout << "[" << severityToString(severity) << "]:" << msg << std::endl;
        }
    }
private:
    const char* severityToString(Severity severity) {
        switch (severity) {
            case Severity::kINTERNAL_ERROR: return "INTERNAL_ERROR";
            case Severity::kERROR: return "ERROR";
            case Severity::kWARNING: return "WARNING";
            case Severity::kINFO: return "INFO";
            case Severity::kVERBOSE: return "VERBOSE";
            default: return "UNKNOWN";
        }
    }
};

class inferenceTool
{
    public:
        inferenceTool() = default;
        explicit inferenceTool(std::string path){};
        ~inferenceTool(){};
        virtual void run(std::vector<float> &input, cv::Mat &output){};
        std::string m_modelPath;
};


