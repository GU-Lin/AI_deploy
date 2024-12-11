#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime_api.h>
#include <fstream>
using namespace nvinfer1;
class Logger : public ILogger {
public:
    void log(ILogger::Severity severity, const char* msg) noexcept override {
        // Only print messages for errors
        if (severity <= ILogger::Severity::kERROR) {
            std::cerr << msg << std::endl;
        }
    }
};
class inferenceTool
{
    public:
        inferenceTool(std::string path);
        ~inferenceTool(){};
        void creatContext(std::string modelPath);
        void run(cv::Mat input, cv::Mat &output);
    private:
        std::string m_modelPath;
        std::shared_ptr<IExecutionContext> Context;
        std::unique_ptr<ICudaEngine> engine;
        std::unique_ptr<IRuntime> runtime;
        Logger gLogger;
};


