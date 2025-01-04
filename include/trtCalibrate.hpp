#include "NvInfer.h"
#include <opencv2/opencv.hpp>
#include "utils.hpp"
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>

class MyInt8Calibrator : public nvinfer1::IInt8EntropyCalibrator2
{
public:
    MyInt8Calibrator(const std::vector<std::string>& calibrationFiles, int inputW, int inputH)
        : mCalibrationFiles(calibrationFiles), mInputW(inputW), mInputH(inputH), mCurrentBatch(0) {
        mInputSize = inputW * inputH * 3 * batch;
        cudaMalloc(&mDeviceInput, mInputSize * sizeof(float));
    }

    ~MyInt8Calibrator() override {
        cudaFree(mDeviceInput);
    }

    int getBatchSize() const noexcept override {
        return 1;
    }

    bool getBatch(void* bindings[], const char* names[], int nbBindings) noexcept override;

    const void* readCalibrationCache(size_t& length) noexcept override ;

    void writeCalibrationCache(const void* cache, size_t length) noexcept override;

private:
    std::vector<std::string> mCalibrationFiles;
    int mInputW, mInputH, mInputSize, mCurrentBatch;
    void* mDeviceInput;
    std::vector<char> mCalibrationCache;
    objectDetection utilsTool;
    int batch = 1;
};
