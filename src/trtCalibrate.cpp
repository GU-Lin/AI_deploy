#include "../include/trtCalibrate.hpp"

bool MyInt8Calibrator::getBatch(void* bindings[], const char* names[], int nbBindings) noexcept
{
    if (mCurrentBatch >= mCalibrationFiles.size())
    {
        return false;  // 所有校準數據處理完畢
    }
    std::vector<float> inputData;
    bool flag = false;
    int count = 0;
    for(int i = 0; i < batch ; i++)
    {
        if(mCurrentBatch+i >= mCalibrationFiles.size())
        {
            flag = true;
            count = i+1;
            break;
        }
        std::string filePath = mCalibrationFiles[mCurrentBatch+i];
        cv::Mat inImg = cv::imread(filePath);  // 使用 OpenCV 讀取圖片
        if (inImg.empty())
        {
            std::cerr << "Failed to load image: " << filePath << std::endl;
            return false;
        }
        cv::Mat outImg;
        std::vector<float> tempData;
        utilsTool.preprocess(inImg,outImg);
        utilsTool.HWC2NormalCHW(outImg, tempData);
        inputData.insert(inputData.end(), tempData.begin(), tempData.end());
        // 3. 將數據拷貝到 GPU
        // cudaMemcpy(mDeviceInput, inputData.data(), mInputSize * sizeof(float), cudaMemcpyHostToDevice);
    }
    if(flag)
    {
        mInputSize = mInputW * mInputH * 3 * count;
    }
    cudaMemcpy(mDeviceInput, inputData.data(), mInputSize * sizeof(float), cudaMemcpyHostToDevice);
    bindings[0] = mDeviceInput;
    mCurrentBatch+=batch;
    return true;
}

const void* MyInt8Calibrator::readCalibrationCache(size_t& length) noexcept
{
    std::ifstream cacheFile("calibration.cache", std::ios::binary);
    if (cacheFile.good()) {
        mCalibrationCache.assign((std::istreambuf_iterator<char>(cacheFile)),
                                    std::istreambuf_iterator<char>());
        length = mCalibrationCache.size();
        return mCalibrationCache.data();
    }
    length = 0;
    return nullptr;
}

void MyInt8Calibrator::writeCalibrationCache(const void* cache, size_t length) noexcept
{
    std::ofstream cacheFile("calibration.cache", std::ios::binary);
    cacheFile.write(reinterpret_cast<const char*>(cache), length);
}
