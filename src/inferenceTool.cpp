#include "../include/inferenceTool.hpp"
inferenceTool::inferenceTool(std::string path)
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
    m_runtime.reset(createInferRuntime(gLogger));
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


int inferenceTool::getIOSize(char const *name)
{
    int temp = 1;
    for(int i = 0; i < m_engine->getTensorShape(name).nbDims; i++)
    {
        temp*=m_engine->getTensorShape(name).d[i];
    }
    return temp;
}

inferenceTool::~inferenceTool()
{
    cudaFree(buffers[0]);
    cudaFree(buffers[1]);
    free(hostData);
    m_context.reset();
    m_runtime.reset();
}

void inferenceTool::run(std::vector<float> &input)
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
    std::cout << "Run done" << std::endl;

}
