#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>


struct PredBox
{
    float x1;
    float y1;
    float x2;
    float y2;
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
        void postprocess();
        void run();
    private:
        // inference engine;
        int i_height;
        int i_width;
        int i_channels;
        int fmap_num;
        int nout;
        float box_threshold;
        float nms_threshold;
        bool autoFlag = true;
        bool scaleFillFlag = false;
        int stride = 32;
};
