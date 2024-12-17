#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <algorithm>

struct PredBox
{
    float boxLeft;
    float boxTop;
    float boxRight;
    float boxDown;
    float score;
    int label;
};

class objectDetection
{
    public:
        objectDetection();
        ~objectDetection();
        void preprocess(cv::Mat input, cv::Mat &output);
        void HWC2NormalCHW(cv::Mat input, std::vector<float> &data);
        float iou(PredBox box1, PredBox box2);
        float areaBox(PredBox box);
        void postprocess();
        void run(std::vector<float> &input);
    private:
        // inference engine;
        int i_height;
        int i_width;
        int i_channels;
        float m_conf_thres = 0.25;
        float m_iou_thres = 0.45;
        bool autoFlag = true;
        bool scaleFillFlag = false;
        int stride = 32;
};
