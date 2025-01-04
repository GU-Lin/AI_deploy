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
#include "inferenceTool.hpp"
class TRTInferenceTool : public inferenceTool
{
    public:
        TRTInferenceTool(std::string path);
        ~TRTInferenceTool();
        void run(std::vector<float> &input, cv::Mat &output) override;
        void setConfig(IInt8EntropyCalibrator2 &Calibrator);
        void buildSerial(std::string input, std::string output);
        void loadOnnxModel(const std::string& onnxFile);
    private:
        int getIOSize(char const *name);

        std::unique_ptr<IExecutionContext> m_context;
        std::shared_ptr<ICudaEngine> m_engine;
        std::unique_ptr<IRuntime> m_runtime;

        // quantization
        std::unique_ptr<IBuilderConfig> m_config;
        std::unique_ptr<IBuilder> m_builder;
        std::unique_ptr<INetworkDefinition> m_network;
        std::unique_ptr<ICudaEngine> m_int8Engine;
        std::unique_ptr<IHostMemory> m_serializedModel;
        int m_inputSize = 0;
        int m_outputSize = 0;
        int m_outputBoxNum = 8400;
        int m_outputClass = 84; // w,y,w,h and 80 class-> 80+4
        Logger gLogger;
        char const* inputName;
        char const* outputName;
        void* buffers[2];
        float* hostData;

};


