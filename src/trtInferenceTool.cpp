#include "../include/trtInferenceTool.hpp"

Logger logger;

TRTInferenceTool::TRTInferenceTool(std::string path)
{
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error opening engine file: " << path << std::endl;
        return ;
    }

    // 讀取文件內容
    file.seekg(0, file.end);
    size_t size = file.tellg();
    file.seekg(0, file.beg);
    std::vector<char> engineData(size);
    file.read(engineData.data(), size);
    file.close();

    // 創建 TensorRT runtime
    m_runtime.reset(createInferRuntime(logger));
    if (!m_runtime) {
        std::cerr << "Failed to create InferRuntime" << std::endl;
        return ;
    }

    // 反序列化模型
    m_engine = std::shared_ptr<ICudaEngine>(
    m_runtime->deserializeCudaEngine(engineData.data(), size), samplesCommon::InferDeleter());
    if (!m_engine) {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        return ;
    }

    m_context.reset(m_engine->createExecutionContext());

    inputName = m_engine->getIOTensorName(0);
    outputName = m_engine->getIOTensorName(1);
    std::cout << "Input  Name : " << inputName << std::endl;
    std::cout << "Output Name : " << outputName << std::endl;

    // CUDA malloc
    m_inputSize = getIOSize(inputName);
    m_outputSize = getIOSize(outputName);
    cudaMalloc(&buffers[0],m_inputSize*sizeof(float));
    cudaMalloc(&buffers[1],m_outputSize*sizeof(float));
    m_context->setTensorAddress(inputName,buffers[0]);
    m_context->setTensorAddress(outputName,buffers[1]);
    hostData = new float[m_outputSize];

    std::cout << "Input size is " << m_inputSize << std::endl;
    std::cout << "Output size is " << m_outputSize << std::endl;

    std::cout << "Load " << path << " successful" << std::endl;
}


int TRTInferenceTool::getIOSize(char const *name)
{
    int temp = 1;
    for(int i = 0; i < m_engine->getTensorShape(name).nbDims; i++)
    {
        std::cout << m_engine->getTensorShape(name).d[i] << ", ";
        temp*=m_engine->getTensorShape(name).d[i];
    }
    return temp;
}

TRTInferenceTool::~TRTInferenceTool()
{
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    free(hostData);
    m_context.reset();
    m_runtime.reset();
}

void TRTInferenceTool::run(std::vector<float> &input, cv::Mat &output)
{
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    // From host to device
    cudaMemcpyAsync(buffers[0], input.data(), m_inputSize * sizeof(float), cudaMemcpyHostToDevice, stream);

    // Enqueue excute
    m_context->enqueueV3(stream);

    // From device to host
    cudaMemcpyAsync(hostData, buffers[1], m_outputSize * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);

    cv::Mat m(m_outputClass,m_outputBoxNum, CV_32FC1, hostData);
    output = m.clone();
}

void TRTInferenceTool::setConfig(IInt8EntropyCalibrator2 &Calibrator)
{
    m_builder.reset(createInferBuilder(logger));
    // m_builder->setInt8Calibrator(&Calibrator);
    m_config.reset(m_builder->createBuilderConfig());
    m_config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,16*(1 << 20));
    m_config->setFlag(nvinfer1::BuilderFlag::kDEBUG);
    m_config->setFlag(BuilderFlag::kINT8); // 啟用 INT8 模式
    // m_config->setAvgTimingIterations(10);
    m_config->setInt8Calibrator(&Calibrator);
}

void TRTInferenceTool::buildSerial(std::string input, std::string output)
{
    m_network.reset(m_builder->createNetworkV2(0));
    loadOnnxModel(input);
    std::cout << "Start reset int8 engine" << std::endl;
    m_int8Engine.reset(m_builder->buildEngineWithConfig(*m_network.get(), *m_config.get()));
    std::cout << "Start convert" << std::endl;
    std::ofstream outputFile(output, std::ios::binary);
    m_serializedModel.reset(m_int8Engine->serialize());
    outputFile.write(reinterpret_cast<const char*>(m_serializedModel->data()),m_serializedModel->size());
    std::cout << "Convert done" << std::endl;
    m_serializedModel.reset();
    outputFile.close();
    m_int8Engine.reset();
    m_network.reset();
    m_config.reset();
    m_builder.reset();
}

void TRTInferenceTool::loadOnnxModel(const std::string& onnxFile) {
    auto parser = nvonnxparser::createParser(*m_network.get(), logger);
    if (!parser->parseFromFile(onnxFile.c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
        throw std::runtime_error("Failed to parse ONNX model: " + onnxFile);
    }

    // 检查是否有输出张量
    if (m_network->getNbOutputs() == 0) {
        throw std::runtime_error("ONNX model must have at least one output.");
    }
    std::cout << "Load onnx sucessful" << std::endl;
    // delete parser;
}
