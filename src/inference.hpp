#include<iostream>
#include<string>
#include<opencv2/opencv.hpp>
class inference
{
    public:
        inference(std::string modelPath);
        virtual ~inference();
        virtual void run(cv::Mat input, cv::Mat &output);
    private:
        std::string m_modelPath;
};
