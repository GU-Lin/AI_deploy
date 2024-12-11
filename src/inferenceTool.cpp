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
    runtime.reset(createInferRuntime(gLogger));
    if (!runtime) {
        std::cerr << "Failed to create InferRuntime" << std::endl;
        return ;
    }

    // 反序列化模型
    engine.reset(runtime->deserializeCudaEngine(engineData.data(), size));
    if (!engine) {
        std::cerr << "Failed to deserialize CUDA engine" << std::endl;
        return ;
    }
    std::cout << "Load " << path << " successful" << std::endl;
}
