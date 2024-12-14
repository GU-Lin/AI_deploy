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
        ~inferenceTool();
        void run(std::vector<float> &input);
    private:
        std::string m_modelPath;
        std::unique_ptr<IExecutionContext> m_context;
        std::shared_ptr<ICudaEngine> m_engine;
        std::unique_ptr<IRuntime> m_runtime;
        int m_inputSize = 0;
        int m_outputSize = 0;
        int getIOSize(char const *name);
        Logger gLogger;
        char const* inputName;
        char const* outputName;
        // const int inputSize;
        // const int outputSize;
        void* buffers[2];
        float* hostData;
};


